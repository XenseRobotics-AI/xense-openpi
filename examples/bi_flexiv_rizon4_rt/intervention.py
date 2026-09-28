"""Pico4 human intervention for BiFlexivRizon4RT inference.

Holding either grip (single-hand takeover) swaps the arm target from policy
output to the Pico4 controller pose; releasing both grips hands control back
to the policy and clears the ActionChunkBroker so the next step re-infers
from the current observation. RLT collection passes a ``TakeoverMotion``: a held
grip then only arms the takeover, which starts once a controller moves.

The module exposes three collaborating pieces:

* ``Pico4InterventionController`` — owns the BiPico4 teleop, decides whether
  intervention is active on each tick, and (on the intervention rising edge)
  syncs the teleop's internal target pose to the live robot TCP so the first
  override frame does not snap the arm.
* ``InterventionEnvironmentWrapper`` — Environment wrapper that polls the
  controller in ``get_observation`` (so ``controller.active`` is fresh before
  the agent runs) and owns teleop disconnect.
* ``InterventionPolicyAgent`` — Agent wrapper that returns the teleop action
  directly while intervention is active (skipping the policy server call) and
  calls ``ActionChunkBroker.reset`` on the release edge.  Stamps every
  returned action dict with ``is_intervention`` so subscribers (e.g. recorders)
  see the same payload that is applied to the robot.
* ``Pico4ButtonMonitor`` — rising-edge watcher for the A/B/X/Y face buttons,
  backed by a 200 Hz sampler thread so short taps are never missed between
  control steps; control-thread callers only drain events and surface faults.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import threading
from typing import TYPE_CHECKING, Any, NamedTuple, override

from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client import action_chunk_broker as _action_chunk_broker
from xense_client.runtime import agent as _agent
from xense_client.runtime import environment as _environment
from xense_client.runtime.agents import policy_agent as _policy_agent

from examples.bi_flexiv_rizon4_rt.button_edges import ButtonEdges
from examples.bi_flexiv_rizon4_rt.button_edges import StreamHealth

if TYPE_CHECKING:
    from lerobot.teleoperators.bi_pico4 import BiPico4

    import examples.bi_flexiv_rizon4_rt.env as _env

logger = get_logger("Pico4Intervention")


# Packing order for the 20D action vector consumed by real_env.step.
# Mirrors _ACTION_LABELS in examples/bi_flexiv_rizon4_rt/main.py and the
# dict packing in examples/bi_flexiv_rizon4_rt/real_env.py.
_ACTION_KEYS_IN_ORDER: tuple[str, ...] = (
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    "left_tcp.r1",
    "left_tcp.r2",
    "left_tcp.r3",
    "left_tcp.r4",
    "left_tcp.r5",
    "left_tcp.r6",
    "right_tcp.x",
    "right_tcp.y",
    "right_tcp.z",
    "right_tcp.r1",
    "right_tcp.r2",
    "right_tcp.r3",
    "right_tcp.r4",
    "right_tcp.r5",
    "right_tcp.r6",
    "left_gripper.pos",
    "right_gripper.pos",
)

# Face buttons watched during inference.  A/B sit on the right controller and
# X/Y on the left, but the XenseVR SDK exposes them as module-level getters on
# the shared ``xrt`` handle, so one table covers both hands.  All four are
# monitored: RLT mode (rlt_mode.py) maps them onto the training signals -
# A = round end / reset confirm, B = window open / success, Y = fail,
# X = discard.
_MONITORED_BUTTONS: tuple[tuple[str, str], ...] = (
    ("A", "get_A_button"),
    ("B", "get_B_button"),
    ("X", "get_X_button"),
    ("Y", "get_Y_button"),
)


class TakeoverMotion(NamedTuple):
    """Controller motion since the grip press that turns the grip into a takeover."""

    position_m: float
    rotation_rad: float


# Gripper convention shared by Pico4 and the robot: 1.0 = open, 0.0 = closed.
_GRIPPER_OPEN_AT = 0.5
# Controller readings kept for the release check, ~0.2 s at the 30 Hz control loop.
_RELEASE_WINDOW_TICKS = 6
_SIDES = ("left", "right")


class _ControllerReading:
    """Raw pose ``[x, y, z, qx, qy, qz, qw]`` (Pico frame) and trigger of both hands."""

    def __init__(self, xrt: Any) -> None:
        self.poses = [
            np.asarray(xrt.get_left_controller_pose(), dtype=np.float64),
            np.asarray(xrt.get_right_controller_pose(), dtype=np.float64),
        ]
        self.triggers = [float(xrt.get_left_trigger()), float(xrt.get_right_trigger())]
        if (
            not all(np.isfinite(pose).all() and pose.shape == (7,) for pose in self.poses)
            or not np.isfinite(self.triggers).all()
        ):
            raise RuntimeError("Invalid Pico4 controller pose/trigger data; refusing to decide a takeover")

    def moved_from(self, origin: _ControllerReading, motion: TakeoverMotion) -> bool:
        """Whether either controller translated or rotated past ``motion`` since ``origin``."""
        for pose, start in zip(self.poses, origin.poses, strict=True):
            if np.linalg.norm(pose[:3] - start[:3]) >= motion.position_m:
                return True
            dot = abs(float(np.dot(pose[3:] / np.linalg.norm(pose[3:]), start[3:] / np.linalg.norm(start[3:]))))
            if 2.0 * np.arccos(min(dot, 1.0)) >= motion.rotation_rad:
                return True
        return False

    def gripper_commands(self, widths: tuple[float, float]) -> list[float]:
        """Gripper targets Pico4 derives from the triggers: ``1 - trigger * width`` per hand."""
        return [1.0 - trigger * width for trigger, width in zip(self.triggers, widths, strict=True)]


class Pico4ButtonMonitor:
    """Rising-edge watcher for the Pico4 A/B/X/Y face buttons.

    A dedicated sampler thread reads the buttons at ``_SAMPLE_INTERVAL_S`` and
    feeds ``ButtonEdges``, so a press is caught no matter how short it is or
    where it lands relative to the (much slower) control loop; with per-step
    polling a tap shorter than the poll interval plus the debounce window
    vanished entirely, which read as intermittent button failure.  Rising
    edges queue the button name for the consumer without blocking the control
    loop; the sampler also keeps the packet-health watchdog fed.  Only the
    rising edge counts, so a held button fires once instead of on every tick.

    Sampler errors (disconnect, invalid read, stale packets) are latched and
    re-raised by ``poll`` on the control thread, preserving the fail-fast
    "abort the round instead of synthesizing a release" semantics.
    """

    # 200 Hz: well above human tap speed, cheap for the local SDK service.
    _SAMPLE_INTERVAL_S = 0.005
    # Shorter than the old 50 ms because the sampler sees every transition;
    # still long enough to reject single-sample glitches.
    _DEBOUNCE_S = 0.03

    def __init__(self, teleop: BiPico4) -> None:
        self._teleop = teleop
        self._edges = ButtonEdges(debounce_s=self._DEBOUNCE_S)
        self._health = StreamHealth()
        self._pending_events: list[str] = []
        self._lock = threading.Lock()
        self._error: RuntimeError | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch the sampler thread (idempotent)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pico4-buttons", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        self._thread = None
        self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self._SAMPLE_INTERVAL_S):
            # Deliberate: BiPico4 exposes no public handle on the SDK module,
            # and the button getters live there. Same access as
            # poll_and_decide's grip reads below.
            xrt = self._teleop._xrt
            if xrt is None:
                self._record_error(RuntimeError("Pico4 disconnected; aborting input processing"))
                continue
            try:
                getter = getattr(xrt, "get_time_stamp_ns", None)
                if getter is None:
                    raise RuntimeError(
                        "Pico4 SDK must expose get_time_stamp_ns for input freshness checking; update the SDK"
                    )
                self._health.update(int(getter()))
                just_pressed: list[str] = []
                for name, getter in _MONITORED_BUTTONS:
                    try:
                        raw = getattr(xrt, getter)()
                        if raw not in (False, True, 0, 1):
                            raise ValueError(f"invalid button value {raw!r}")
                    except Exception as e:
                        raise RuntimeError(
                            f"Pico4 input unavailable ({name}); abort round instead of synthesizing a release"
                        ) from e
                    if self._edges.update(name, bool(raw)):
                        just_pressed.append(name)
            except Exception as e:
                self._record_error(e)
                continue
            if just_pressed:
                with self._lock:
                    self._pending_events.extend(just_pressed)

    def _record_error(self, error: RuntimeError) -> None:
        with self._lock:
            self._edges.reset()
            self._pending_events.clear()
            self._error = error

    def poll(self) -> None:
        """Surface sampler errors on the control thread (no-op when healthy).

        The sampler queues events in the background, so control-step callers
        no longer need to drive sampling; they only need the fail-fast error
        path.  The error is delivered once — a persistent fault re-latches on
        the next sample.
        """
        self.start()  # Lazy start for callers that never saw an explicit start().
        with self._lock:
            error = self._error
            self._error = None
        if error is not None:
            raise error

    def consume_events(self) -> list[str]:
        with self._lock:
            events = self._pending_events
            self._pending_events = []
        return events


class Pico4InterventionController:
    """Owns the BiPico4 teleop and tracks intervention mode each tick."""

    def __init__(
        self, teleop: BiPico4, base_env: _env.BiFlexivRizon4RTEnvironment, takeover_motion: TakeoverMotion | None = None
    ) -> None:
        """``takeover_motion=None`` takes over the moment a grip is held.

        With a ``TakeoverMotion`` (RLT collection) a held grip only arms the
        takeover: the policy keeps running until a controller moves that far
        from where it was when the grip was pressed.
        """
        self._teleop = teleop
        self._base_env = base_env
        self._takeover_motion = takeover_motion
        self._armed_at: _ControllerReading | None = None
        # Per hand (left, right): grip down this tick, driving last tick, and
        # the opening a non-driving hand holds instead of its trigger's.
        self._grip_held: tuple[bool, bool] = (False, False)
        self._driving = [False, False]
        self._held_gripper: list[float | None] = [None, None]
        # Readings while taken over, for the "was the hand still at release" check.
        self._recent: deque[_ControllerReading] = deque(maxlen=_RELEASE_WINDOW_TICKS)
        self._active = False
        self._was_active = False
        self._pending_release = False
        self._buttons = Pico4ButtonMonitor(teleop)

    def set_takeover_motion(self, motion: TakeoverMotion | None) -> None:
        self._takeover_motion = motion

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        """Connect the BiPico4 to the already-running robot.

        Must be called after the underlying robot is connected (so the initial
        TCP pose is readable).  pre_init overlaps XenseVR SDK setup with robot
        init; here the robot is already up, so we just call connect directly.
        """
        left_pose, right_pose = self._base_env._env.robot.get_current_tcp_pose_quat()
        logger.info("Connecting BiPico4 teleop for intervention...")
        self._teleop.connect(
            calibrate=False,
            left_tcp_pose_quat=left_pose,
            right_tcp_pose_quat=right_pose,
        )
        self._buttons.start()  # Sampler thread seeds packet health and held-button baselines itself.
        logger.info("BiPico4 intervention armed (hold either grip to take over).")

    def poll_buttons(self) -> None:
        """Surface sampler errors outside the control loop (e.g. the reset gate).

        The sampler thread queues face-button events continuously, so this no
        longer drives sampling; it only re-raises latched sampler faults.
        """
        self._buttons.poll()

    def consume_button_events(self) -> list[str]:
        return self._buttons.consume_events()

    def poll_and_decide(self, gripper_command: Sequence[float] | None = None) -> bool:
        """Refresh intervention state; return True if we should override policy.

        Called once per control step from the environment wrapper's
        ``get_observation``.  On the intervention rising edge (non-active →
        active), resyncs the teleop's target pose to the live robot TCP so
        the first override frame does not snap the arm.  ``reset_to_pose``
        logs at INFO level, so doing it only on the edge (rather than every
        non-intervention frame) keeps the log stream sane.

        ``Pico4ButtonMonitor.poll`` runs first so latched sampler faults abort
        this tick before any grip decision or action; button events themselves
        queue continuously in the background sampler thread.
        """
        self._buttons.poll()

        xrt = self._teleop._xrt
        if xrt is None:
            raise RuntimeError("Pico4 disconnected; policy handback requires a new session")

        left_grip = float(xrt.get_left_grip())
        right_grip = float(xrt.get_right_grip())
        if not np.isfinite([left_grip, right_grip]).all() or not (0 <= left_grip <= 1 and 0 <= right_grip <= 1):
            raise RuntimeError("Invalid Pico4 grip data; aborting instead of resuming policy")
        cfg = self._teleop.config
        # Apply BiPico4's hysteresis thresholds per side, combined with OR:
        # either hand alone takes over, and control returns to the policy only
        # once both grips are released.  The unheld arm stays frozen on its
        # takeover-synced target because each Pico4 instance gates on its own
        # grip.  Using raw grips (rather than each Pico4._enabled) keeps the
        # decision independent of whether get_action has already been called
        # this frame.
        enter_hi = cfg.grip_enable_threshold
        exit_lo = cfg.grip_disable_threshold
        threshold = exit_lo if self._active or self._armed_at is not None else enter_hi
        self._grip_held = (left_grip > threshold, right_grip > threshold)
        held = any(self._grip_held)
        new_active = self._decide_takeover(xrt, held=held)

        was_active = self._active
        rising_edge = not was_active and new_active

        if rising_edge:
            # Sync _target_pos/_quat to the live TCP *before* BiPico4.get_action
            # is called for the first override frame. get_action sees _enabled
            # rising → sets _ref_pos from the current controller position →
            # delta = 0 on frame 1 → output ≈ _target_pos ≈ current TCP, so
            # the handoff is continuous.
            try:
                left_pose, right_pose = self._base_env._env.robot.get_current_tcp_pose_quat()
                self._teleop.reset_to_pose(
                    left_pose[:7],
                    right_pose[:7],
                    float(left_pose[7]),
                    float(right_pose[7]),
                )
            except Exception as e:
                raise RuntimeError("Pico4 handoff failed; refusing to execute the old policy target") from e

        self._was_active = was_active
        self._active = new_active
        self._sync_gripper_hold(xrt, gripper_command)

        if was_active and not new_active:
            self._pending_release = True
            logger.info("Intervention released (both grips < disable threshold).")
            if self._takeover_motion is not None:
                self._warn_if_not_still(_ControllerReading(xrt))
        elif rising_edge:
            logger.info(f"Intervention engaged — policy paused (grips L={left_grip:.2f} R={right_grip:.2f}).")

        return self._active

    def _decide_takeover(self, xrt: Any, *, held: bool) -> bool:
        """Whether the operator drives the arm this tick, given the grip state.

        With a ``TakeoverMotion`` the takeover needs the controller to move;
        a gripper that disagrees with the robot's no longer holds it back, it
        only gets reported (``_sync_gripper_hold``).
        """
        if not held:
            self._armed_at = None
            return False
        if self._takeover_motion is None:
            return True
        reading = _ControllerReading(xrt)
        if self._active:
            self._recent.append(reading)
            return True
        if self._armed_at is None:
            # Grip pressed: remember where the controllers are; the policy runs on.
            self._armed_at = reading
            logger.info("Intervention armed — policy continues until the controller moves.")
            return False
        if not reading.moved_from(self._armed_at, self._takeover_motion):
            return False
        self._recent.clear()
        self._recent.append(reading)
        return True

    def _sync_gripper_hold(self, xrt: Any, gripper_command: Sequence[float] | None) -> None:
        """Decide, per hand, whether its gripper follows the trigger this tick.

        Only the hand whose grip is held is driving; BiPico4 nevertheless maps
        every trigger onto its gripper unconditionally, so a one-handed
        takeover used to open the *other* hand's gripper and drop whatever it
        was holding.  A hand that is not driving latches the opening the robot
        was last commanded and keeps commanding exactly that until its own
        grip goes down — latched once rather than tracked live, because the
        measured opening of a gripper holding an object sits at the object's
        width and re-commanding it would loosen the grip step by step.

        A hand that *starts* driving (this takeover, or a second grip pressed
        during one) snaps its gripper to the trigger; that is how a takeover
        works now, so it is reported rather than prevented.
        """
        if not self._active:
            self._held_gripper = [None, None]
            self._driving = [False, False]
            return
        robot: list[float] | None = None
        for index, held in enumerate(self._grip_held):
            if held:
                if not self._driving[index]:
                    robot = self._robot_grippers(gripper_command) if robot is None else robot
                    self._warn_gripper_jump(xrt, index, robot[index])
                self._held_gripper[index] = None
            elif self._held_gripper[index] is None:
                robot = self._robot_grippers(gripper_command) if robot is None else robot
                self._held_gripper[index] = robot[index]
            self._driving[index] = held

    def _warn_gripper_jump(self, xrt: Any, index: int, target: float) -> None:
        """Report this hand's gripper crossing open/closed as it starts driving."""
        cfg = self._teleop.config
        value = _ControllerReading(xrt).gripper_commands((cfg.left_gripper_width, cfg.right_gripper_width))[index]
        if (value >= _GRIPPER_OPEN_AT) != (target >= _GRIPPER_OPEN_AT):
            logger.warn(
                f"Takeover: {_SIDES[index]} gripper jumps {target:.2f} -> {value:.2f} (robot -> trigger); anything it was holding is released."
            )

    def _robot_grippers(self, gripper_command: Sequence[float] | None) -> list[float]:
        """The openings the robot is holding: its last command, or the measurement before any."""
        return [float(value) for value in gripper_command] if gripper_command is not None else self._measured_grippers()

    def _measured_grippers(self) -> list[float]:
        """Robot gripper readings, used before any gripper command was sent."""
        left_pose, right_pose = self._base_env._env.robot.get_current_tcp_pose_quat()
        return [float(left_pose[7]), float(right_pose[7])]

    def _warn_if_not_still(self, release: _ControllerReading) -> None:
        """Warn when the hand was still moving or switching a gripper as the grip let go."""
        motion = self._takeover_motion
        window = [*self._recent, release]
        self._recent.clear()
        if motion is None or len(window) < 2:
            return
        cfg = self._teleop.config
        widths = (cfg.left_gripper_width, cfg.right_gripper_width)
        if release.moved_from(window[0], motion):
            logger.warn(
                "Intervention released while the controller was still moving; hold the hand still before letting go."
            )
        states = {tuple(value >= _GRIPPER_OPEN_AT for value in reading.gripper_commands(widths)) for reading in window}
        if len(states) > 1:
            logger.warn(
                "Intervention released while a gripper was switching open/closed; settle the trigger before letting go."
            )

    def consume_release_event(self) -> bool:
        """Return True exactly once after each intervention→policy transition."""
        if self._pending_release:
            self._pending_release = False
            return True
        return False

    def get_override_action(self) -> np.ndarray:
        """Sample the BiPico4 and pack into a 20D float32 vector.

        Key ordering follows _ACTION_KEYS_IN_ORDER, matching the 20D action
        vector consumed by BiFlexivRizon4RTRealEnv.step.  The last two dims
        are the grippers: a hand that is not driving keeps the opening
        ``_sync_gripper_hold`` latched for it instead of BiPico4's
        trigger-derived one.
        """
        raw = self._teleop.get_action()
        action = np.asarray([float(raw[k]) for k in _ACTION_KEYS_IN_ORDER], dtype=np.float32)
        for index, hold in enumerate(self._held_gripper):
            if hold is not None:
                action[index - 2] = hold
        if not np.isfinite(action).all():
            raise RuntimeError("Non-finite Pico4 action; refusing to send robot target")
        return action

    def reset_for_new_episode(self) -> None:
        """Clear intervention state at episode boundary.

        Ensures the next ``poll_and_decide`` treats a held grip as a fresh
        rising edge, which forces ``reset_to_pose`` to resync the teleop's
        target to wherever the robot landed after ``env.reset``.  Without
        this, a user who keeps the grips held across an episode reset would
        see the arm snap back to the pre-reset target pose on the first
        post-reset tick.
        """
        self._active = False
        self._was_active = False
        self._pending_release = False
        self._armed_at = None
        self._grip_held = (False, False)
        self._driving = [False, False]
        self._held_gripper = [None, None]
        self._recent.clear()

    def disconnect(self) -> None:
        self._buttons.stop()
        try:
            self._teleop.disconnect()
        except Exception as e:
            logger.warn(f"Error disconnecting BiPico4: {e}")


class InterventionEnvironmentWrapper(_environment.Environment):
    """Wraps an Environment to poll the Pico4 controller before the agent
    runs and to own teleop disconnect.  Action swapping happens in
    ``InterventionPolicyAgent`` so runtime subscribers (e.g. recorders)
    see the same action dict that is applied to the robot.
    """

    def __init__(
        self,
        wrapped_env: _environment.Environment,
        controller: Pico4InterventionController,
    ) -> None:
        self._wrapped_env = wrapped_env
        self._controller = controller

    @override
    def reset(self) -> None:
        self._wrapped_env.reset()
        # Force a rising edge on the first post-reset poll so the teleop
        # target gets resynced to the robot's new home pose. Otherwise a
        # user holding the grips across the reset would see the arm snap
        # back to the pre-reset target.
        self._controller.reset_for_new_episode()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        # Poll grips before the agent runs so controller.active is fresh when
        # InterventionPolicyAgent.get_action reads it on the next line of the
        # runtime loop.
        self._controller.poll_and_decide()
        return self._wrapped_env.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        self._wrapped_env.apply_action(action)

    def disconnect(self) -> None:
        self._controller.disconnect()
        inner_disconnect = getattr(self._wrapped_env, "disconnect", None)
        if callable(inner_disconnect):
            inner_disconnect()


class InterventionPolicyAgent(_agent.Agent):
    """Wraps a PolicyAgent: returns the teleop action (and skips the policy
    server call) while intervention is active; clears the chunk broker on
    the release edge so the next step triggers a fresh inference.

    The returned dict always carries ``is_intervention`` so recorders and
    other subscribers can distinguish human-commanded steps from policy
    steps.
    """

    def __init__(
        self,
        inner_agent: _policy_agent.PolicyAgent,
        controller: Pico4InterventionController,
        broker: _action_chunk_broker.ActionChunkBroker,
    ) -> None:
        self._inner = inner_agent
        self._controller = controller
        self._broker = broker

    @override
    def get_action(self, observation: dict) -> dict:
        if self._controller.active:
            return {
                "actions": self._controller.get_override_action(),
                "is_intervention": True,
            }

        if self._controller.consume_release_event():
            logger.info("Clearing ActionChunkBroker cache (intervention released).")
            self._broker.reset()

        button_events = self._controller.consume_button_events()
        if button_events:
            observation = {**observation, "__button_events__": button_events}

        action = self._inner.get_action(observation)
        # Copy to avoid mutating a dict that may be cached internally by the
        # chunk broker, and stamp the flag for downstream subscribers.
        out: dict[str, Any] = {**action, "is_intervention": False}
        return out

    @override
    def reset(self) -> None:
        self._inner.reset()

    @override
    def warmup(self, observation: dict) -> None:
        self._inner.warmup(observation)
