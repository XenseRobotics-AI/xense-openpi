import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class WujiWeightLoader(WeightLoader):
    """Loads a full checkpoint except the action projections, which keep the model's random init.

    The released pi0/pi05 checkpoints are trained with action_dim=32. The Tianji/Wuji
    robot needs action_dim=58, which changes the shape of both `action_in_proj`
    (action_dim -> width) and `action_out_proj` (width -> action_dim). Those weights are
    dropped from the checkpoint and handed back unloaded, so `train.py` fills them from
    the model's own initializer. Every other weight - VLM, action expert, time MLPs - is
    loaded unchanged.

    Any other shape mismatch between checkpoint and model raises, so a checkpoint that
    disagrees in more than the action width is not loaded silently.

    Compatible with:
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/pi05_base/params"
    """

    params_path: str
    # Flat param paths (joined with "/") that are re-initialised instead of loaded.
    reinit_regex: str = "action_(in|out)_proj/.*"

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")

        pattern = re.compile(self.reinit_regex)
        reinit_keys = sorted(k for k in flat_ref if pattern.fullmatch(k))
        if not reinit_keys:
            raise ValueError(f"reinit_regex {self.reinit_regex!r} matches no model parameter.")
        for key in reinit_keys:
            flat_loaded.pop(key, None)

        mismatched = [
            f"{k}: checkpoint {v.shape} vs model {flat_ref[k].shape}"
            for k, v in flat_loaded.items()
            if k in flat_ref and v.shape != flat_ref[k].shape
        ]
        if mismatched:
            raise ValueError(
                f"Checkpoint {self.params_path} does not match the model outside {self.reinit_regex!r}:\n  "
                + "\n  ".join(mismatched)
            )

        logger.info("Re-initialising %d params instead of loading them: %s", len(reinit_keys), reinit_keys)
        return _merge_params(
            flax.traverse_util.unflatten_dict(flat_loaded, sep="/"),
            params,
            missing_regex=f"(?:{self.reinit_regex})|(?:.*lora.*)",
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz",
            gs={"token": "anon"},
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
