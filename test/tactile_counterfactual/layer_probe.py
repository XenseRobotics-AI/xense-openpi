"""Shared machinery of the two layer-wise tactile probes (step 1 of
docs/action-conditioned-tactile-pretraining.md):

* ``scripts/probe_future_tactile_layers.py``      linear (ridge) probe per layer, picks ``m``
* ``scripts/probe_tactile_sensitivity_layers.py`` causal sensitivity per layer (section 6.1)

Both probes run the *training* forward (prefix + suffix in one pass, no KV cache) on a
batch of real observations at a fixed flow time ``tau`` and fixed noise, and read the
action expert's residual stream after every block through ``return_suffix_hidden``.
Everything here is a plain function of one ``Pi0TactileFastVit`` checkpoint; nothing is
trained.

Pieces:

``load_setup``            train config + norm stats + dataset + model, from a config name
                          and a checkpoint step directory (the counterfactual runner's loaders).
``FutureTactileStore``    read-only view of a label store from
                          ``scripts/compute_tactile_future_labels.py``: future targets per
                          ``(episode, frame)`` and a contact proxy from the 16x16 pixel field.
``sample_frames``         frames drawn across episodes so that every batch holds distinct
                          episodes (the roll partner must come from somewhere else), with the
                          contact / non-contact mix stratified through the store's proxy.
``iter_batches``          decodes the sampled frames with dataloader workers.
``LayerForward``          the jitted forward: tactile tokens, the action-position residual
                          stream of every block, the pooled VLM output and ``v_t``.
``make_variant``          the perturbations of section 6.1 (five core variants) plus the
                          extra counterfactual conditions of section 6.2.
``sensitivity``           ``1 - cos_cent`` per sample, centred on the batch mean of ``real``.
``fit_ridge`` / ``evaluate_ridge``  closed-form ridge with the penalty picked on a
                          held-out episode split.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import json
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_tactile_fastvit import Pi0TactileFastVit
from openpi.models.pi0_tactile_fastvit_config import Pi0TactileFastVitConfig
from openpi.shared import nnx_utils
import openpi.training.config as _config
from openpi.transforms import InjectTactileFutureLabels
from test.tactile_counterfactual import runner as _runner
from test.tactile_counterfactual.counterfactual import make_counterfactual_observation
from test.tactile_counterfactual.dataset_index import ProbeDataset

logger = logging.getLogger("tactile_layer_probe")

# Flow times the probes sweep. tau = 1 is pure noise in the action tokens (no action
# conditioning); tau = 0.25 is an almost clean action chunk.
TAUS = (0.25, 0.5, 0.75, 1.0)

# Section 6.1: the five variants every run computes ...
CORE_VARIANTS = ("real", "null", "tac-shuffle", "vl-swap", "pad-pert")
# ... and the extra counterfactual conditions of section 6.2 (opt-in).
EXTRA_VARIANTS = ("tac-zero", "pad-swap", "tac-timeshift")
ALL_VARIANTS = CORE_VARIANTS + EXTRA_VARIANTS

CONTACT_PROXIES = ("delta", "state")


# --------------------------------------------------------------------------- #
# Model / dataset loading                                                     #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class ProbeSetup:
    config_name: str
    checkpoint_dir: pathlib.Path
    train_config: _config.TrainConfig
    data_config: _config.DataConfig
    model: Pi0TactileFastVit
    dataset: ProbeDataset

    @property
    def model_config(self) -> Pi0TactileFastVitConfig:
        return self.train_config.model  # type: ignore[return-value]

    @property
    def tactile_keys(self) -> tuple[str, ...]:
        return tuple(self.model_config.tactile_image_keys)


def load_setup(
    config_name: str,
    checkpoint_dir: str | pathlib.Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    root: str | None = None,
    cudnn_attention: bool = False,
) -> ProbeSetup:
    """Resolve the train config, norm stats, dataset and model for one checkpoint.

    The LTP head (``tactile_future_layer``) is switched off: the probes read the residual
    stream directly and a step-1 checkpoint has no head parameters. cuDNN attention is
    off by default -- the probes compare bit patterns between forwards, and the explicit
    attention path is the one every GPU runs identically.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_dir}")
    train_config = _config.get_config(config_name)
    model_config = train_config.model
    if not isinstance(model_config, Pi0TactileFastVitConfig):
        raise TypeError(
            f"{config_name} is a {type(model_config).__name__}; the layer probes need Pi0TactileFastVitConfig"
        )
    overrides: dict[str, Any] = {}
    if model_config.use_cudnn_attention != cudnn_attention:
        overrides["use_cudnn_attention"] = cudnn_attention
    if model_config.tactile_future_layer is not None:
        overrides["tactile_future_layer"] = None
    if overrides:
        logger.info("model config overrides for the probe: %s", overrides)
        train_config = dataclasses.replace(train_config, model=dataclasses.replace(model_config, **overrides))

    data_config = _runner.resolve_data_config(train_config, None, checkpoint_dir)  # type: ignore[arg-type]
    dataset = ProbeDataset(
        repo_id or data_config.repo_id,
        data_config,
        action_horizon=train_config.model.action_horizon,
        revision=revision,
        root=root,
    )
    model, _ = _runner.load_model(train_config, checkpoint_dir)
    return ProbeSetup(
        config_name=config_name,
        checkpoint_dir=checkpoint_dir,
        train_config=train_config,
        data_config=data_config,
        model=model,  # type: ignore[arg-type]
        dataset=dataset,
    )


# --------------------------------------------------------------------------- #
# Label store                                                                 #
# --------------------------------------------------------------------------- #


class FutureTactileStore:
    """Read-only view of one label store directory.

    ``targets(episode, frame)`` returns the same ``(future_tactile_z, future_tactile_mask)``
    the training transform injects. ``contact_scores`` turns the stored 16x16 pixel field
    into the contact proxy of section 6.1: ``"delta"`` is ``||y_delta||`` at the first
    horizon (change over the next ``k`` frames), ``"state"`` is the distance to the
    episode's first frame (how far the gel is from its resting look right now). Both are
    normalised by the store's per-pad RMS so ~1 is "typical".
    """

    def __init__(self, labels_dir: str | pathlib.Path, horizons: Sequence[int], target: str = "pixel_delta") -> None:
        self.labels_dir = pathlib.Path(labels_dir).expanduser()
        self.meta = json.loads((self.labels_dir / "meta.json").read_text())
        self.offsets = np.load(self.labels_dir / "episode_offsets.npy")
        self.horizons = tuple(int(k) for k in horizons)
        self.target = target
        self._inject = InjectTactileFutureLabels(labels_dir=str(self.labels_dir), horizons=self.horizons, target=target)
        self._field = np.load(self.labels_dir / "pixel_field.npy", mmap_mode="r")
        self._rms = {int(k): np.asarray(v, dtype=np.float32) for k, v in self.meta["y_delta_rms"].items()}

    @property
    def num_episodes(self) -> int:
        return len(self.offsets) - 1

    @property
    def num_pads(self) -> int:
        return int(self.meta["num_pads"])

    def episode_length(self, episode: int) -> int:
        return int(self.offsets[episode + 1] - self.offsets[episode])

    def target_dim(self) -> int:
        if self.target == "latent":
            return int(self.meta["pca_dim"])
        return int(np.prod(self._field.shape[2:]))

    def targets(self, episode: int, frame: int) -> tuple[np.ndarray, np.ndarray]:
        out = self._inject({"episode_index": episode, "frame_index": frame})["aux_targets"]
        return out["future_tactile_z"], out["future_tactile_mask"]

    def contact_scores(self, episodes: np.ndarray, frames: np.ndarray, proxy: str = "delta") -> np.ndarray:
        """Vectorised proxy for many ``(episode, frame)`` pairs -> float32 [n]."""
        if proxy not in CONTACT_PROXIES:
            raise ValueError(f"proxy must be one of {CONTACT_PROXIES}, got {proxy!r}")
        episodes = np.asarray(episodes, dtype=np.int64)
        frames = np.asarray(frames, dtype=np.int64)
        start = self.offsets[episodes]
        length = self.offsets[episodes + 1] - start
        if np.any(frames < 0) or np.any(frames >= length):
            raise IndexError("frame out of range for its episode")
        current = self._field[start + frames].astype(np.float32) / 255.0  # [n, S, h, w, 3]
        if proxy == "delta":
            k = self.horizons[0]
            other_idx = start + np.minimum(frames + k, length - 1)
            rms = self._rms[k]
        else:
            k = self.horizons[-1]
            other_idx = start
            rms = self._rms[k]
        other = self._field[other_idx].astype(np.float32) / 255.0
        diff = (other - current) / rms[None, :, None, None, None]
        return np.sqrt(np.mean(np.square(diff), axis=(1, 2, 3, 4))).astype(np.float32)


# --------------------------------------------------------------------------- #
# Frame sampling                                                              #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class FrameRefs:
    """The probe's sample set, in batch order: batch ``i`` is rows ``i*B:(i+1)*B``."""

    episode: np.ndarray  # int64 [N]
    frame: np.ndarray  # int64 [N]
    batch_size: int
    contact: np.ndarray | None = None  # bool [N], None without a label store
    score: np.ndarray | None = None  # float32 [N]
    contact_threshold: float | None = None
    contact_proxy: str | None = None

    def __len__(self) -> int:
        return int(self.episode.shape[0])

    @property
    def num_batches(self) -> int:
        return (len(self) + self.batch_size - 1) // self.batch_size

    def batch(self, i: int) -> slice:
        return slice(i * self.batch_size, min((i + 1) * self.batch_size, len(self)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode.tolist(),
            "frame": self.frame.tolist(),
            "batch_size": self.batch_size,
            "contact": None if self.contact is None else self.contact.tolist(),
            "score": None if self.score is None else [float(s) for s in self.score],
            "contact_threshold": self.contact_threshold,
            "contact_proxy": self.contact_proxy,
        }


def sample_frames(
    dataset: ProbeDataset,
    *,
    num_frames: int,
    batch_size: int,
    rng: np.random.Generator,
    min_tail: int = 0,
    store: FutureTactileStore | None = None,
    contact_quantile: float = 0.5,
    contact_proxy: str = "delta",
    pool_size: int = 4096,
    max_tries: int = 64,
    episodes: Sequence[int] | None = None,
) -> FrameRefs:
    """Draw ``num_frames`` frames such that every batch holds distinct episodes.

    ``min_tail`` keeps ``frame + min_tail < episode length`` (clean action window, future
    labels, time-shift donors). With a store, the batch alternates contact and
    non-contact slots: a frame counts as contact when its proxy is above the
    ``contact_quantile`` of a random pool of frames drawn over the eligible episodes.
    """
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2: the roll partner has to be another sample")
    candidates = list(episodes) if episodes is not None else list(dataset.episodes)
    eligible = [ep for ep in candidates if dataset.episode_length(ep) > min_tail + 1]
    if store is not None:
        eligible = [ep for ep in eligible if ep < store.num_episodes]
    if not eligible:
        raise ValueError(f"no episode is longer than min_tail={min_tail}")
    if len(eligible) < batch_size:
        logger.warning(
            "only %d eligible episodes for batch_size=%d; batches will repeat episodes and the roll "
            "partner may come from the same episode (S_tac under-estimated)",
            len(eligible),
            batch_size,
        )

    def draw_frame(ep: int) -> int:
        return int(rng.integers(0, dataset.episode_length(ep) - min_tail))

    threshold: float | None = None
    if store is not None:
        pool_eps = rng.choice(eligible, size=pool_size, replace=True)
        pool_frames = np.array([draw_frame(int(ep)) for ep in pool_eps])
        pool_scores = store.contact_scores(pool_eps, pool_frames, contact_proxy)
        threshold = float(np.quantile(pool_scores, contact_quantile))
        logger.info(
            "contact proxy %r over %d pooled frames: median %.3f, threshold (q=%.2f) %.3f",
            contact_proxy,
            pool_size,
            float(np.median(pool_scores)),
            contact_quantile,
            threshold,
        )

    eps_out: list[int] = []
    frames_out: list[int] = []
    scores_out: list[float] = []
    num_batches = (num_frames + batch_size - 1) // batch_size
    for b in range(num_batches):
        n_this = min(batch_size, num_frames - b * batch_size)
        order = rng.permutation(len(eligible))
        chosen = [eligible[order[i % len(eligible)]] for i in range(n_this)]
        for slot, ep in enumerate(chosen):
            if store is None:
                frame = draw_frame(ep)
                score = np.nan
            else:
                want_contact = slot % 2 == 0
                frame, score = draw_frame(ep), np.nan
                for _ in range(max_tries):
                    frame = draw_frame(ep)
                    score = float(store.contact_scores(np.array([ep]), np.array([frame]), contact_proxy)[0])
                    if (score >= threshold) == want_contact:
                        break
            if not dataset.has_sample(ep, frame):
                raise RuntimeError(f"(episode {ep}, frame {frame}) is not in the dataset index")
            eps_out.append(int(ep))
            frames_out.append(int(frame))
            scores_out.append(score)

    scores = np.asarray(scores_out, dtype=np.float32)
    contact = None if store is None else scores >= threshold
    if contact is not None:
        logger.info(
            "sampled %d frames, %d contact / %d non-contact", len(eps_out), int(contact.sum()), int((~contact).sum())
        )
    return FrameRefs(
        episode=np.asarray(eps_out, dtype=np.int64),
        frame=np.asarray(frames_out, dtype=np.int64),
        batch_size=batch_size,
        contact=contact,
        score=None if store is None else scores,
        contact_threshold=threshold,
        contact_proxy=None if store is None else contact_proxy,
    )


def split_episodes(episodes: np.ndarray, val_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean [N] mask: True where the frame belongs to a held-out (validation) episode."""
    unique = np.unique(episodes)
    n_val = max(1, round(len(unique) * val_fraction))
    if n_val >= len(unique):
        raise ValueError(f"val_fraction={val_fraction} leaves no training episode out of {len(unique)}")
    val_eps = set(rng.choice(unique, size=n_val, replace=False).tolist())
    return np.array([int(ep) in val_eps for ep in episodes], dtype=bool)


# --------------------------------------------------------------------------- #
# Batch loading                                                               #
# --------------------------------------------------------------------------- #

_OBS_KEYS = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")


@dataclasses.dataclass
class Batch:
    index: np.ndarray  # positions in the FrameRefs
    episode: np.ndarray
    frame: np.ndarray
    observation: _model.Observation
    actions: np.ndarray  # float32 [B, ah, ad], normalised like training


class _FrameDataset:
    """torch map-style dataset over the sampled (episode, frame) pairs."""

    def __init__(self, dataset: ProbeDataset, episode: np.ndarray, frame: np.ndarray) -> None:
        self._dataset = dataset
        self._episode = episode
        self._frame = frame

    def __len__(self) -> int:
        return int(self._episode.shape[0])

    def __getitem__(self, i: int) -> dict[str, Any]:
        sample = self._dataset.get_sample(int(self._episode[i]), int(self._frame[i]))
        out = {k: sample[k] for k in _OBS_KEYS if k in sample}
        out["actions"] = np.asarray(sample["actions"], dtype=np.float32)
        out["_index"] = np.asarray(i, dtype=np.int64)
        return out


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *samples)


def observation_from_batch(batch: dict[str, Any]) -> _model.Observation:
    data = {k: jax.tree.map(jnp.asarray, batch[k]) for k in _OBS_KEYS if k in batch}
    return _model.Observation.from_dict(data)


def iter_batches(
    dataset: ProbeDataset,
    refs: FrameRefs,
    *,
    num_workers: int = 0,
    episode: np.ndarray | None = None,
    frame: np.ndarray | None = None,
) -> Iterator[Batch]:
    """Decode the frames of ``refs`` batch by batch (optionally other frames in the same order,
    e.g. the time-shifted donors: pass ``episode`` / ``frame``)."""
    import multiprocessing

    import torch.utils.data as torch_data

    episode = refs.episode if episode is None else np.asarray(episode, dtype=np.int64)
    frame = refs.frame if frame is None else np.asarray(frame, dtype=np.int64)
    loader = torch_data.DataLoader(
        _FrameDataset(dataset, episode, frame),
        batch_size=refs.batch_size,
        shuffle=False,
        num_workers=num_workers,
        # Same choice as openpi.training.data_loader: JAX is multithreaded in the parent,
        # so forked workers may deadlock; spawned ones re-import and pickle the dataset.
        multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
        collate_fn=_collate,
        drop_last=False,
    )
    for raw in loader:
        index = np.asarray(raw.pop("_index"))
        yield Batch(
            index=index,
            episode=episode[index],
            frame=frame[index],
            observation=observation_from_batch(raw),
            actions=np.asarray(raw["actions"], dtype=np.float32),
        )


# --------------------------------------------------------------------------- #
# Forward                                                                     #
# --------------------------------------------------------------------------- #


def layer_names(depth: int) -> list[str]:
    """``l0`` = suffix input tokens (before block 1), ``l1..lD`` = output of block 1..D."""
    return [f"l{i}" for i in range(depth + 1)]


class LayerForward:
    """Jitted training-style forward at a fixed ``(tau, noise)`` with per-block outputs.

    Returns (all float32, host-convertible):

    ``tactile_tokens``  [B, S_tac, D]      ``tactile_proj`` output as it enters the suffix
    ``action_stream``   [D+1, B, ah, D]    residual stream at the action positions: entry 0
                                           is the suffix input (``action_in_proj`` of
                                           ``x_t``), entry ``l`` the output of block ``l``
    ``prefix_pooled``   [B, D_vlm]         masked mean of the PaliGemma output tokens
    ``v_t``             [B, ah, ad]        the velocity prediction
    ``u_t``             [B, ah, ad]        the flow target ``noise - actions`` (for the loss)
    """

    def __init__(self, model: Pi0TactileFastVit) -> None:
        if not isinstance(model, Pi0TactileFastVit):
            raise TypeError(f"LayerForward needs a Pi0TactileFastVit, got {type(model).__name__}")
        self._model = model
        self._graphdef, self._state = nnx.split(model)
        self._fn = jax.jit(self._fun)

    @property
    def layer_names(self) -> list[str]:
        return layer_names_from_model(self._model)

    def _fun(self, state, obs: _model.Observation, actions, noise, tau):
        module: Pi0TactileFastVit = nnx.merge(self._graphdef, state)
        obs = module._preprocess_observation(None, obs, train=False)
        batch = obs.state.shape[0]
        time = jnp.broadcast_to(tau.astype(jnp.float32), (batch,))
        x_t = tau * noise + (1.0 - tau) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar = module.embed_prefix(obs)
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = module.embed_suffix(obs, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _, hidden = module.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            return_suffix_hidden=True,
        )
        ah = module.action_horizon
        n_tac = module._num_tactile
        v_t = module.action_out_proj(suffix_out[:, -ah:])
        weight = prefix_mask[:, :, None].astype(jnp.float32)
        prefix_pooled = jnp.sum(prefix_out.astype(jnp.float32) * weight, axis=1) / jnp.maximum(
            jnp.sum(weight, axis=1), 1.0
        )
        action_stream = jnp.concatenate(
            [suffix_tokens[None, :, -ah:].astype(jnp.float32), hidden[:, :, -ah:].astype(jnp.float32)], axis=0
        )
        return {
            "tactile_tokens": suffix_tokens[:, :n_tac].astype(jnp.float32),
            "action_stream": action_stream,
            "prefix_pooled": prefix_pooled,
            "v_t": v_t.astype(jnp.float32),
            "u_t": u_t.astype(jnp.float32),
        }

    def __call__(self, obs: _model.Observation, actions, noise, tau: float) -> dict[str, jax.Array]:
        return self._fn(
            self._state,
            obs,
            jnp.asarray(actions, dtype=jnp.float32),
            jnp.asarray(noise, dtype=jnp.float32),
            jnp.asarray(tau, dtype=jnp.float32),
        )


def layer_names_from_model(model: Pi0TactileFastVit) -> list[str]:
    """``l0..lD`` for the action expert (the last Gemma expert) of ``model``."""
    depth = int(model.PaliGemma.llm.module.configs[-1].depth)  # ToNNX keeps the linen module
    return layer_names(depth)


class ChunkSampler:
    """The production sampler with caller-fixed noise: final action chunk per variant."""

    def __init__(self, model: Pi0TactileFastVit, *, rtc: bool) -> None:
        fn = model.training_time_rtc_sample_actions if rtc else model.sample_actions
        self._fn = nnx_utils.module_jit(fn)
        self.mode = "rtc" if rtc else "standard"

    def __call__(self, obs: _model.Observation, noise, num_steps: int) -> jax.Array:
        return self._fn(jax.random.key(0), obs, num_steps=num_steps, noise=jnp.asarray(noise, dtype=jnp.float32))


def fixed_noise(seed: int, num: int, action_horizon: int, action_dim: int) -> np.ndarray:
    """One noise chunk per sampled frame, shared by every tau and every variant."""
    return np.asarray(jax.random.normal(jax.random.key(seed), (num, action_horizon, action_dim)), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Variants                                                                    #
# --------------------------------------------------------------------------- #


def _roll(x, shift: int = -1):
    return jnp.roll(jnp.asarray(x), shift, axis=0)


def make_variant(
    obs: _model.Observation,
    name: str,
    tactile_keys: Sequence[str],
    *,
    pad_extra: int = 2,
    donor: _model.Observation | None = None,
) -> _model.Observation:
    """Build one perturbed observation. Never mutates ``obs``.

    real          nothing
    null          tactile ``image_mask=False`` (the tokens drop out of attention)
    tac-shuffle   tactile images + masks rolled by one inside the batch; the rest untouched
    vl-swap       the three RGB views (+ masks) and the tokenized prompt rolled by one;
                  tactile untouched. NOTE: in pi05 the discretised state lives inside the
                  prompt, so this also swaps the state (vision + proprioception sensitivity).
    pad-pert      the last ``pad_extra`` valid prompt tokens masked out. The pi05 prompt ends
                  in ``;\\nAction: ``, so a small ``pad_extra`` removes only formatting tokens.
    tac-zero      tactile images set to 0 (mid grey in [-1, 1]), masks kept
    pad-swap      left and right jaw swapped: keys[0] <-> keys[2], keys[1] <-> keys[3]
    tac-timeshift tactile images + masks taken from ``donor`` (same episode, other frame)
    """
    tactile_keys = tuple(tactile_keys)
    if name == "real":
        return obs
    if name == "null":
        masks = dict(obs.image_masks)
        for key in tactile_keys:
            masks[key] = jnp.zeros_like(jnp.asarray(masks[key]), dtype=jnp.bool_)
        return dataclasses.replace(obs, image_masks=masks)
    if name == "tac-shuffle":
        images, masks = dict(obs.images), dict(obs.image_masks)
        for key in tactile_keys:
            images[key] = _roll(images[key])
            masks[key] = _roll(masks[key])
        return dataclasses.replace(obs, images=images, image_masks=masks)
    if name == "vl-swap":
        images, masks = dict(obs.images), dict(obs.image_masks)
        for key in images:
            if key not in tactile_keys:
                images[key] = _roll(images[key])
                masks[key] = _roll(masks[key])
        kwargs: dict[str, Any] = {"images": images, "image_masks": masks}
        if obs.tokenized_prompt is not None:
            kwargs["tokenized_prompt"] = _roll(obs.tokenized_prompt)
            kwargs["tokenized_prompt_mask"] = _roll(obs.tokenized_prompt_mask)
        return dataclasses.replace(obs, **kwargs)
    if name == "pad-pert":
        if obs.tokenized_prompt_mask is None:
            raise ValueError("pad-pert needs a tokenized prompt")
        mask = jnp.asarray(obs.tokenized_prompt_mask)
        lengths = jnp.sum(mask.astype(jnp.int32), axis=1)
        keep = jnp.maximum(lengths - pad_extra, 1)
        idx = jnp.arange(mask.shape[1])[None, :]
        return dataclasses.replace(obs, tokenized_prompt_mask=mask & (idx < keep[:, None]))
    if name == "tac-zero":
        images = dict(obs.images)
        for key in tactile_keys:
            images[key] = jnp.zeros_like(jnp.asarray(images[key]))
        return dataclasses.replace(obs, images=images)
    if name == "pad-swap":
        if len(tactile_keys) != 4:
            raise ValueError(f"pad-swap assumes 4 tactile keys (left top/bottom, right top/bottom), got {tactile_keys}")
        images, masks = dict(obs.images), dict(obs.image_masks)
        for a, b in ((0, 2), (1, 3)):
            ka, kb = tactile_keys[a], tactile_keys[b]
            images[ka], images[kb] = obs.images[kb], obs.images[ka]
            masks[ka], masks[kb] = obs.image_masks[kb], obs.image_masks[ka]
        return dataclasses.replace(obs, images=images, image_masks=masks)
    if name == "tac-timeshift":
        if donor is None:
            raise ValueError("tac-timeshift needs the donor observation (same episodes, shifted frames)")
        return make_counterfactual_observation(obs, donor, tactile_keys)
    raise ValueError(f"unknown variant {name!r}; known: {ALL_VARIANTS}")


def assert_variant_differs(obs: _model.Observation, variant: _model.Observation, tactile_keys: Sequence[str]) -> None:
    """Self-check: a tactile perturbation must change the tactile pixels of (almost) every row."""
    for key in tactile_keys:
        a = np.asarray(obs.images[key])
        b = np.asarray(variant.images[key])
        same = np.all(a.reshape(a.shape[0], -1) == b.reshape(b.shape[0], -1), axis=1)
        if same.all():
            raise AssertionError(f"variant leaves tactile image {key!r} unchanged in every row")


# --------------------------------------------------------------------------- #
# Sensitivity metrics                                                         #
# --------------------------------------------------------------------------- #


def _flat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1)


def centered_cosine(real: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Per-sample cosine after subtracting the batch mean of ``real`` from both sides.

    Inputs ``[B, ...]``. The shared constant every sample carries (learned biases, the
    time embedding) pushes raw cosine to ~1 and hides the per-sample variation; centring
    is what makes the number readable.
    """
    a, b = _flat(real), _flat(other)
    mu = a.mean(axis=0, keepdims=True)
    a, b = a - mu, b - mu
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def sensitivity(real: np.ndarray, other: np.ndarray) -> np.ndarray:
    """``1 - cos_cent(real, other)`` per sample; ``[B]`` for ``[B, ...]`` inputs."""
    return 1.0 - centered_cosine(real, other)


def cross_sample_sensitivity(real: np.ndarray) -> np.ndarray:
    """``S_x``: ``1 - cos_cent(real[i], real[i+1])``, the natural variation between samples."""
    return sensitivity(real, np.roll(np.asarray(real), -1, axis=0))


def layered_sensitivity(real: np.ndarray, other: np.ndarray) -> np.ndarray:
    """For ``[L, B, ...]`` inputs: ``[L, B]`` per-layer, per-sample sensitivities."""
    return np.stack([sensitivity(real[i], other[i]) for i in range(real.shape[0])])


def layered_cross_sample(real: np.ndarray) -> np.ndarray:
    return np.stack([cross_sample_sensitivity(real[i]) for i in range(real.shape[0])])


def summarize(values: np.ndarray, subset: np.ndarray | None = None) -> dict[str, float | int] | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if subset is not None:
        values = values[np.asarray(subset, dtype=bool).reshape(-1)]
    if values.size == 0:
        return None
    return {"mean": float(values.mean()), "std": float(values.std()), "n": int(values.size)}


# --------------------------------------------------------------------------- #
# Ridge                                                                       #
# --------------------------------------------------------------------------- #

# Relative to trace(X^T X) / d; the top end lets a layer with nothing to say fall back to
# (almost) the mean predictor instead of over-fitting 1024+ dims to ~1.5k rows.
DEFAULT_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


@dataclasses.dataclass
class RidgeModel:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    weight: np.ndarray  # [d, T]
    alpha: float
    lam: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std
        return xs @ self.weight + self.y_mean


def _standardize(x_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x_tr.mean(axis=0)
    x_std = x_tr.std(axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    return x_mean, x_std


def fit_ridge(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> tuple[RidgeModel, dict[float, float]]:
    """Closed-form ridge on standardised features; ``lambda = alpha * trace(X^T X) / d``.

    The penalty is chosen on the validation rows (overall normalised MSE). Primal form when
    ``d <= n``, dual (kernel) form otherwise, so wide feature sets stay cheap.
    Returns the refitted best model and the per-alpha validation nMSE.
    """
    x_tr = np.asarray(x_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64).reshape(x_tr.shape[0], -1)
    x_va = np.asarray(x_va, dtype=np.float64)
    y_va = np.asarray(y_va, dtype=np.float64).reshape(x_va.shape[0], -1)
    x_mean, x_std = _standardize(x_tr)
    y_mean = y_tr.mean(axis=0)
    xs = (x_tr - x_mean) / x_std
    ys = y_tr - y_mean
    xs_va = (x_va - x_mean) / x_std
    n, d = xs.shape
    scale = float(np.sum(np.square(xs)) / d)
    baseline = float(np.sum(np.square(y_va - y_mean)))

    primal = d <= n
    if primal:
        eigvals, eigvecs = np.linalg.eigh(xs.T @ xs)
        proj = eigvecs.T @ (xs.T @ ys)
    else:
        eigvals, eigvecs = np.linalg.eigh(xs @ xs.T)
        proj = eigvecs.T @ ys
    eigvals = np.clip(eigvals, 0.0, None)

    def weight_for(lam: float) -> np.ndarray:
        solved = eigvecs @ (proj / (eigvals + lam)[:, None])
        return solved if primal else xs.T @ solved

    scores: dict[float, float] = {}
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in alphas:
        lam = float(alpha) * scale
        w = weight_for(lam)
        nmse = float(np.sum(np.square(xs_va @ w + y_mean - y_va)) / max(baseline, 1e-12))
        scores[float(alpha)] = nmse
        if best is None or nmse < best[1]:
            best = (float(alpha), nmse, w)
    assert best is not None
    return RidgeModel(
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, weight=best[2], alpha=best[0], lam=best[0] * scale
    ), scores


def evaluate_ridge(model: RidgeModel, x: np.ndarray, y: np.ndarray, target_shape: Sequence[int]) -> dict[str, Any]:
    """Normalised MSE (1 - R^2 against the train-mean predictor) overall, per horizon, per pad.

    ``target_shape`` is ``(K, S, Z)``; the columns of ``y`` are that shape flattened.
    """
    y = np.asarray(y, dtype=np.float64).reshape(x.shape[0], -1)
    pred = model.predict(x)
    mse = np.sum(np.square(pred - y), axis=0).reshape(target_shape)
    base = np.sum(np.square(model.y_mean - y), axis=0).reshape(target_shape)

    def ratio(a, b):
        return float(np.sum(a) / max(float(np.sum(b)), 1e-12))

    return {
        "nmse": ratio(mse, base),
        "per_horizon": [ratio(mse[k], base[k]) for k in range(target_shape[0])],
        "per_pad": [ratio(mse[:, s], base[:, s]) for s in range(target_shape[1])],
        "n": int(x.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Reporting helpers                                                           #
# --------------------------------------------------------------------------- #


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _cell(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "nan"
        return f"{v:.4f}" if abs(v) >= 1e-2 or v == 0 else f"{v:.2e}"
    return str(v)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return obj
