"""Pi0 model with tactile image support.

This extends the base Pi0 model to support additional tactile image inputs
while preserving the ability to load pretrained weights for the visual branch.
"""

import einops
import flax.nnx as nnx
from flax.nnx import bridge as nnx_bridge
import jax.numpy as jnp

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_tactile_config
from openpi.models import siglip as _siglip
from openpi.shared import array_typing as at

# Visual camera names (compatible with pretrained weights)
VISUAL_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

# Tactile sensor names (new inputs)
TACTILE_CAMERAS = ("left_tactile_0_rgb", "right_tactile_0_rgb")


class Pi0Tactile(nnx.Module):
    """Pi0 model with tactile image support.

    Architecture:
    - Visual branch (3 cameras): Uses pretrained SigLIP encoder
    - Tactile branch (2 sensors): Uses plain vit vision encoder (trained from scratch)
    - Both branches produce tokens that are concatenated before feeding to the LLM
    """

    def __init__(self, config: pi0_tactile_config.Pi0TactileConfig, rngs: nnx.Rngs):
        # Initialize parent class (this creates the visual encoder and LLM)
        super().__init__(config, rngs)

        # Store config for later use
        self.config = config

        # Create a separate SigLIP encoder for tactile images
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        tactile_img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # Initialize with a fake tactile image
        fake_tactile_image = next(img for name, img in config.fake_obs().images.items() if name in TACTILE_CAMERAS)
        tactile_img.lazy_init(fake_tactile_image, train=False, rngs=rngs)

        # Store tactile encoder
        self.tactile_img = tactile_img

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed prefix with both visual and tactile images.

        This overrides the parent method to handle tactile images separately.
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # Embed visual images using pretrained encoder
        for name in obs.images:
            if name not in VISUAL_CAMERAS:
                continue  # Skip tactile images for now

            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        # Embed tactile images using new encoder
        for name in obs.images:
            if name not in TACTILE_CAMERAS:
                continue  # Skip visual images

            tactile_tokens, _ = self.tactile_img(obs.images[name], train=False)
            tokens.append(tactile_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=tactile_tokens.shape[1],
                )
            )
            ar_mask += [False] * tactile_tokens.shape[1]

        # Add language tokens (same as parent)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        # Concatenate all tokens
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask
