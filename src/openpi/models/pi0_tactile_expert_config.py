"""JAX pi05 with a separate tactile expert for the final flow steps."""

import dataclasses

import flax.nnx as nnx

from openpi.models import gemma
from openpi.models.pi0_tactile_fastvit_config import Pi0TactileFastVitConfig


@dataclasses.dataclass(frozen=True)
class Pi0TactileExpertConfig(Pi0TactileFastVitConfig):
    pi05: bool = True
    tactile_expert_variant: gemma.Variant = "gemma_300m"
    cascade_total_steps: int = 10
    cascade_split_step: int = 6
    tactile_loss_weight: float = 1.0
    tactile_dropout: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or self.enable_training_time_rtc:
            raise ValueError("The tactile cascade supports only pi05 standard denoising (no RTC)")
        if self.cascade_total_steps <= 0 or not 0 <= self.cascade_split_step <= self.cascade_total_steps:
            raise ValueError("Require 0 <= cascade_split_step <= cascade_total_steps, with total_steps > 0")
        if not 0 <= self.tactile_dropout < 1 or not self.tactile_loss_weight >= 0:
            raise ValueError("Require 0 <= tactile_dropout < 1 and tactile_loss_weight >= 0")
        variants = (self.paligemma_variant, self.action_expert_variant, self.tactile_expert_variant)
        if any("lora" in variant for variant in variants):
            raise ValueError("The tactile cascade supports full training only, not LoRA")
        configs, _ = self.expert_configs()
        for field in ("depth", "num_heads", "num_kv_heads", "head_dim"):
            if len({getattr(config, field) for config in configs}) != 1:
                raise ValueError(f"All cascade experts must have the same {field}")
        if not self.tactile_image_keys or len(set(self.tactile_image_keys)) != len(self.tactile_image_keys):
            raise ValueError("tactile_image_keys must be nonempty and unique")

    def expert_configs(self):
        configs, adarms = super().expert_configs()
        return [*configs, gemma.get_config(self.tactile_expert_variant)], [*adarms, True]

    def create(self, rng):
        from openpi.models.pi0_tactile_expert import Pi0TactileExpert

        return Pi0TactileExpert(self, rngs=nnx.Rngs(rng))
