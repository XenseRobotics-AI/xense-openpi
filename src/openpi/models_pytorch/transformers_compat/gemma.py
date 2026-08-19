"""Pi0-specific Gemma subclasses targeting transformers==5.3.0.

Re-creates the deltas that used to live in ``transformers_replace/models/gemma/``
by subclassing the upstream 5.3.0 classes:

- ``use_adarms`` / ``adarms_cond_dim`` config fields
- ``GemmaRMSNorm`` accepting an optional conditioning tensor, returning
  ``(output, gate_or_None)`` so decoder layers can apply gated residuals
- Per-layer ``adarms_cond`` threaded through ``GemmaDecoderLayer`` /
  ``GemmaModel`` / ``GemmaForCausalLM``
- ``GemmaAttention`` preserving the "read-but-don't-append" KV-cache branch Pi0
  relies on in the suffix-only forward path
- Skipping the ``hidden_states *= sqrt(hidden_size)`` embedding normalizer the
  original patch removed
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma import modeling_gemma as _mg
from transformers.models.gemma.configuration_gemma import GemmaConfig


def gated_residual(x: torch.Tensor | None, y: torch.Tensor | None, gate: torch.Tensor | None) -> torch.Tensor | None:
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


class PiGemmaConfig(GemmaConfig):
    model_type = "gemma"

    def __init__(self, use_adarms: bool = False, adarms_cond_dim: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.use_adarms = use_adarms
        if use_adarms and adarms_cond_dim is None:
            adarms_cond_dim = self.hidden_size
        self.adarms_cond_dim = adarms_cond_dim


class PiGemmaRMSNorm(_mg.GemmaRMSNorm):
    """RMSNorm that optionally produces scale/shift/gate from a condition vector.

    When ``cond_dim`` is None this is byte-compatible with the upstream norm's
    output (wrapped in a ``(out, None)`` tuple); when provided, the forward
    pass matches the original Pi0 ``transformers_replace`` behaviour.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__(dim, eps=eps)
        self.cond_dim = cond_dim
        if cond_dim is not None:
            # The adaptive path uses ``self.dense`` and ignores the stock gain
            # ``self.weight``; we keep the inherited weight only because
            # ``GemmaPreTrainedModel._init_weights`` unconditionally references it.
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.dense = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        dtype = x.dtype
        normed = self._norm(x.float())
        if cond is None or self.dense is None:
            out = normed * (1.0 + self.weight.float())
            return out.to(dtype), None

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond last-dim {self.cond_dim}, got {cond.shape[-1]}")
        modulation = self.dense(cond)
        # If cond is per-batch (b, cond_dim) but x is per-token (b, seq, dim),
        # broadcast modulation across the sequence dim. When cond is already
        # per-token (b, seq, cond_dim) — e.g. RTC training-time per-action
        # timesteps — modulation is already (b, seq, dim*3) and broadcasts
        # naturally.
        if x.dim() == 3 and modulation.dim() == 2:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        out = normed * (1.0 + scale.to(torch.float32)) + shift.to(torch.float32)
        return out.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        base = f"{tuple(self.weight.shape)}, eps={self.eps}"
        if self.dense is not None:
            base += f", adaptive=True, cond_dim={self.cond_dim}"
        return base


class PiGemmaAttention(_mg.GemmaAttention):
    """Preserves Pi0's dual-mode KV cache handling.

    Upstream 5.3 always calls ``past_key_values.update(...)``. Pi0's suffix-only
    forward path passes an existing prefix cache together with ``use_cache=False``
    and expects the new K/V tensors to be *concatenated for attention* but
    *not* written back into the cache (so the prefix cache stays reusable
    across samples).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            # The parameter is Optional only to match the HF attention signature;
            # this layer never computes rotary embeddings itself.
            raise ValueError("position_embeddings is required by this attention layer.")
        cos, sin = position_embeddings
        query_states, key_states = _mg.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if use_cache:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                # 5.3 DynamicCache exposes keys/values per layer via ``.layers[i]``;
                # earlier versions exposed a ``(key, value)`` tuple via ``__getitem__``.
                if hasattr(past_key_values, "layers"):
                    layer_cache = past_key_values.layers[self.layer_idx]
                    cached_keys, cached_values = layer_cache.keys, layer_cache.values
                else:
                    layer_cache = past_key_values[self.layer_idx]
                    cached_keys, cached_values = layer_cache[0], layer_cache[1]
                key_states = torch.cat([cached_keys, key_states], dim=2)
                value_states = torch.cat([cached_values, value_states], dim=2)

        attention_interface: Callable = _mg.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, _mg.eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PiGemmaDecoderLayer(_mg.GemmaDecoderLayer):
    def __init__(self, config: PiGemmaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = PiGemmaAttention(config=config, layer_idx=layer_idx)
        cond_dim = config.adarms_cond_dim if getattr(config, "use_adarms", False) else None
        self.input_layernorm = PiGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
        self.post_attention_layernorm = PiGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, adarms_cond)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = gated_residual(residual, hidden_states, gate)

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = gated_residual(residual, hidden_states, gate)
        return hidden_states


class PiGemmaModel(_mg.GemmaModel):
    config_class = PiGemmaConfig

    def __init__(self, config: PiGemmaConfig):
        super().__init__(config)
        # Replace upstream layers/norm with Pi0 variants. We leave embed_tokens
        # and rotary_emb as-is from the parent class.
        self.layers = nn.ModuleList(
            [PiGemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        cond_dim = config.adarms_cond_dim if getattr(config, "use_adarms", False) else None
        self.norm = PiGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _mg.create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        if len(self.layers) > 0 and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # Pi0 deliberately skips the ``hidden_states * sqrt(hidden_size)`` scaling
        # that upstream Gemma applies before the decoder stack.

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, adarms_cond)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class PiGemmaForCausalLM(_mg.GemmaForCausalLM):
    config_class = PiGemmaConfig

    def __init__(self, config: PiGemmaConfig):
        super().__init__(config)
        self.model = PiGemmaModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            adarms_cond=adarms_cond,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
        )
