from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias, override

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import torch
from xense_client import base_policy as _base_policy

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._tactile_audit_done = False

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            # Mirror the JAX path: prefer the RTC sampler when the model was
            # trained with ``enable_training_time_rtc=True``.
            if getattr(model, "_enable_training_time_rtc", False) and hasattr(
                model, "training_time_rtc_sample_actions"
            ):
                logging.info("Using training_time_rtc_sample_actions for PyTorch inference")
                self._sample_actions = model.training_time_rtc_sample_actions
            else:
                self._sample_actions = model.sample_actions
        else:
            # JAX model setup - choose sample method based on training_time_rtc config
            if getattr(model, "_enable_training_time_rtc", False) and hasattr(
                model, "training_time_rtc_sample_actions"
            ):
                logging.info("Using training_time_rtc_sample_actions for inference")
                self._sample_actions = nnx_utils.module_jit(model.training_time_rtc_sample_actions)
            else:
                self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, **kwargs) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        self._audit_tactile_inputs_once(inputs)
        rtc_inputs = None

        # RTC prefixes must live in the same action space the model was trained on.
        # For delta/normalized policies, the client sends the already-executed
        # absolute actions back as `prev_chunk_left_over`. We transform those
        # actions using the *same* input pipeline as normal training/inference,
        # but without disturbing the real observation batch used for sampling.
        if "prev_chunk_left_over" in kwargs and kwargs["prev_chunk_left_over"] is not None:
            rtc_obs = jax.tree.map(lambda x: x, obs)
            # Some websocket/msgpack paths return read-only NumPy views. The
            # input transform stack may mutate `actions` in place
            # (e.g. DeltaActions), so RTC prefixes must be copied to a
            # writable array before reusing the training-time preprocessing.
            rtc_obs["actions"] = np.array(kwargs["prev_chunk_left_over"], copy=True)
            rtc_inputs = self._input_transform(rtc_obs)

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
            if rtc_inputs is not None:
                rtc_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], rtc_inputs)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device
            if rtc_inputs is not None:
                rtc_inputs = jax.tree.map(
                    lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                    rtc_inputs,
                )

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)

        # Process additional kwargs (e.g. RTC parameters)
        for k, v in kwargs.items():
            # Convert scalars to arrays to avoid recompilation in JAX
            if isinstance(v, (int, float)):
                if self._is_pytorch_model:
                    v = torch.tensor(v).to(self._pytorch_device)
                else:
                    v = jnp.asarray(v)

            if isinstance(v, np.ndarray):
                if self._is_pytorch_model:
                    v = torch.from_numpy(v).to(self._pytorch_device)
                    if v.ndim == 2:  # Assuming (chunk, dim) -> (1, chunk, dim)
                        v = v.unsqueeze(0)
                else:
                    v = jnp.asarray(v)
                    if v.ndim == 2:  # Assuming (chunk, dim) -> (1, chunk, dim)
                        v = v[np.newaxis, ...]

            if k == "prev_chunk_left_over" and rtc_inputs is not None:
                v = rtc_inputs["actions"]
            sample_kwargs[k] = v

        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # Keep a copy of the original actions before output transforms
        actions_original = outputs["actions"]

        outputs = self._output_transform(outputs)
        outputs["actions_original"] = actions_original
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _audit_tactile_inputs_once(self, inputs: dict) -> None:
        if self._tactile_audit_done:
            return

        images = inputs.get("image")
        image_masks = inputs.get("image_mask")
        if not isinstance(images, dict) or not isinstance(image_masks, dict):
            return

        tactile_keys = ("tactile_0_rgb", "tactile_1_rgb", "tactile_2_rgb", "tactile_3_rgb")
        if not any(key in images for key in tactile_keys):
            return

        self._tactile_audit_done = True
        missing_images = [key for key in tactile_keys if key not in images]
        missing_masks = [key for key in tactile_keys if key not in image_masks]
        if missing_images or missing_masks:
            raise RuntimeError(
                "Tactile policy audit failed after input transforms: "
                f"missing image keys={missing_images}, missing mask keys={missing_masks}, "
                f"available image keys={tuple(images.keys())}"
            )

        for key in tactile_keys:
            image = np.asarray(images[key])
            mask = bool(np.asarray(image_masks[key]))
            if image.ndim != 3 or image.shape[-1] != 3:
                raise RuntimeError(f"Tactile policy audit expected HWC image for {key}, got shape={image.shape}")
            if image.shape[:2] != _model.IMAGE_RESOLUTION:
                raise RuntimeError(
                    f"Tactile policy audit expected {key} resolution {_model.IMAGE_RESOLUTION}, got {image.shape[:2]}"
                )
            if not mask:
                raise RuntimeError(f"Tactile policy audit found {key} image_mask=False")
            if not np.isfinite(image).all():
                raise RuntimeError(f"Tactile policy audit found NaN or Inf in {key}")

            image_min = float(image.min())
            image_max = float(image.max())
            image_mean = float(image.mean())
            image_std = float(image.std())
            if np.issubdtype(image.dtype, np.floating) and (image_min < -1.05 or image_max > 1.05):
                raise RuntimeError(
                    f"Tactile policy audit found {key} outside [-1, 1]: min={image_min:.4f}, max={image_max:.4f}"
                )
            if image_std <= 1e-6:
                raise RuntimeError(f"Tactile policy audit found constant image for {key}: std={image_std:.6g}")

            logging.info(
                "[TACTILE SERVER] %s shape=%s dtype=%s mask=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                key,
                image.shape,
                image.dtype,
                mask,
                image_min,
                image_max,
                image_mean,
                image_std,
            )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
