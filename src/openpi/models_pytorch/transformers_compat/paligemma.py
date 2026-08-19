"""Pi0-specific PaliGemma subclasses targeting transformers==5.3.0.

Upstream 5.3 ``PaliGemmaModel`` instantiates its language model via
``AutoModel.from_config(config.text_config)``, which would resolve to the stock
``GemmaModel``.  We override ``__init__`` to construct our ``PiGemmaModel``
instead, and override ``get_image_features`` to skip the ``/ sqrt(hidden_size)``
scaling that Pi0 has always removed.

We also expose ``language_model`` as an attribute on the top-level
``PiPaliGemmaForConditionalGeneration`` so existing openpi call-sites that
referenced ``self.paligemma.language_model`` keep working after the 5.3
restructuring (which moved it under ``self.model.language_model``).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.paligemma import modeling_paligemma as _mp

from openpi.models_pytorch.transformers_compat.gemma import PiGemmaModel


def _siglip_encoder_dtype_hook(module, args, kwargs):
    """Cast the encoder input to the encoder's weight dtype.

    Pi0 keeps ``patch_embedding`` / ``position_embedding`` in float32 while the
    rest of the SigLIP encoder runs in bfloat16, so the embeddings' fp32 output
    needs to be downcast before the first layer norm. The stock 5.3 HF forward
    calls ``encoder(inputs_embeds=...)`` as a kwarg, so we intercept with
    ``with_kwargs=True``.
    """
    if len(module.layers) == 0:
        return None
    target_dtype = module.layers[0].self_attn.q_proj.weight.dtype

    def _cast(t):
        if isinstance(t, torch.Tensor) and t.dtype != target_dtype:
            return t.to(target_dtype)
        return t

    new_args = tuple(_cast(a) for a in args)
    new_kwargs = {k: _cast(v) for k, v in kwargs.items()}
    return new_args, new_kwargs


class PiPaliGemmaModel(_mp.PaliGemmaModel):
    def __init__(self, config):
        # Let the parent build vision_tower, projector, and the upstream
        # language_model; then swap the language model for our Pi0 subclass
        # (keeping the already-built vision tower intact).
        super().__init__(config)
        # ``config.text_config`` was already promoted to our ``PiGemmaConfig``
        # by the caller before ``PiPaliGemmaModel`` is constructed, so the
        # extra ``use_adarms`` / ``adarms_cond_dim`` fields are visible here.
        self.language_model = PiGemmaModel(config.text_config)
        self.vision_tower.vision_model.encoder.register_forward_pre_hook(_siglip_encoder_dtype_hook, with_kwargs=True)
        self.post_init()

    def get_image_features(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Return raw projected image features as a tensor (no sqrt scaling).

        Tolerates both ``[B, C, H, W]`` (the SigLIP conv expects this) and
        ``[B, H, W, C]`` (what JAX-style ``FakeDataset`` and unprocessed
        observations produce); the latter is permuted on the fly.
        """
        if pixel_values.dim() == 4 and pixel_values.shape[1] != 3 and pixel_values.shape[-1] == 3:
            pixel_values = pixel_values.permute(0, 3, 1, 2).contiguous()
        image_outputs = self.vision_tower(pixel_values)
        selected = image_outputs.last_hidden_state
        return self.multi_modal_projector(selected)


class PiPaliGemmaForConditionalGeneration(_mp.PaliGemmaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = PiPaliGemmaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    # Back-compat aliases for openpi's gemma_pytorch.py / pi0_pytorch.py, which
    # still reach into ``paligemma.<vision_tower|language_model>`` rather than
    # the 5.3 layout where these live under ``paligemma.model.*``.
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower
