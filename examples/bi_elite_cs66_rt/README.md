# BiElite CS66 RT — Real Robot Inference

Dual-arm Elite CS66 inference client using the OpenPI policy server. Trained with
`LeRobotBiEliteDataConfig` (see `configs/_examples/pi05_base_bi_elite_*.yaml`).

State/action are 20D and **per-side-grouped**, the driver's own feature order:

```
[left_tcp.x/y/z/r1-r6 (0-8), left_gripper.pos (9),
 right_tcp.x/y/z/r1-r6 (10-18), right_gripper.pos (19)]
```

This differs from the BiFlexiv example (both grippers last). TCP poses are in the
driver's world frame — see [`recipes/README.md`](recipes/README.md) for why the
recipe's mount rotations must match the training data.

## Usage

### Terminal 1: policy server (GPU machine)

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_bi_elite_egg_carton_to_tray_1012_h100 \
    --policy.dir=<checkpoint_dir>
```

### Terminal 2: robot client (`lerobot-xense` conda environment)

Always start with a dry run — the arms connect and go to the start pose, but
policy actions are only printed:

```bash
python -m examples.bi_elite_cs66_rt.main --args.run dry-run --args.host <server>
```

Then for real:

```bash
python -m examples.bi_elite_cs66_rt.main \
    --args.robot-recipe diagonal-08-taccap --args.host <server> --args.port 8000
```

Useful flags (see `--help` for all):

| Flag | Effect |
|---|---|
| `--args.rtc-enabled` | Real-time chunking; the prefix is re-based on the server |
| `--args.action-hz 60` | Decoupled runtime: actions at 60 Hz, obs at `runtime_hz` |
| `--args.max-lin-speed 0.15` / `--args.max-ang-speed 0.8` | Tighter servo-loop speed caps than the recipe's |
| `--args.no-go-to-start` | Start from wherever the arms are |
| `--args.visualize True` / `--args.visualize False` | Override the recipe's optional Xense Motion Viewer switch |

Ctrl+C stops the runtime and disconnects the robot; a second Ctrl+C forces exit.

Bench-08 enables native geometric collision checking by default, independently
of the Viewer. See [collision settings](recipes/README.md#collision-checking)
and the comments in [the recipe](recipes/diagonal-08-taccap.yaml).

## Optional Xense Motion Viewer visualization

The display is off by default. It adds read-only chunk/progress subscriptions
and an independent sampler of the existing Elite SDK RTSI/gripper streams.
Image processing, Rerun logging and rendering run in a spawned XMV process.
It does not read policy observations to obtain joints, open another robot
connection, call IK, or change action queue replacement/blending.

Install the [optional viewer dependency](../../README.md#optional-xense-motion-viewer)
in the robot client's `lerobot-xense` environment first. XMV pins Rerun `0.26.2`,
within lerobot's supported range.

### Enable or disable

Enable visualization for the dry run above with an explicit boolean:

```bash
python -m examples.bi_elite_cs66_rt.main \
    --args.run dry-run --args.host <server> --args.visualize True --args.enable-tactile
```

`--args.enable-tactile` uses the existing device setting to enable the four
RECTIFY RGB tactile streams. The display flag does not enable devices or change
connection checks. Omit the tactile flag to keep those devices off; their panels
then show that no observations are available. The policy still receives only
head and wrist images. Dry run connects hardware and can reset the robot; it is
not a hardware-free test.

Use `--args.visualize False` to force it off for one launch. For repeated runs,
set `visualize: true` or `visualize: false` in the run YAML selected by
`--args.run`. The precedence is **CLI > run YAML > recipe**; an omitted or `null`
run setting follows the recipe. These overrides only change whether the display
is enabled; its other settings still come from the recipe.

To enable it by default for a bench, edit the existing top-level block in the
robot recipe (beside `robot:`, not inside it):

```yaml
visualization:
  enabled: true
  display_hz: 30
  images_enabled: true
  image_hz: 10
  image_max_edge: 640
  jpeg_quality: 85
  image_stale_after: 1.0
  spawn_viewer: true
  viewer_port: 9877
  save_path: null  # or recordings/elite-live.rrd
```

For recording without a viewer window, keep `enabled: true`, set
`spawn_viewer: false` and set `save_path: recordings/elite-live.rrd`. Relative
recording paths are resolved from the working directory.

The shipped recipe uses `enabled: false`. When off, unused display settings are
ignored; no subscriptions, state sampler, image transfer thread or display process are created, and
XMV/Rerun are not imported. Switching visualization off leaves
`robot.native_collision_profile` unchanged; see
[collision checking](recipes/README.md#collision-checking).

### Display behavior

The display process receives data through bounded channels and has no robot
object or SDK callbacks. A display startup failure is logged and inference
continues with visualization disabled. SDK state sampling remains independent
of `get_observation()`.

Seven image panels show the head camera, two wrists, and four tactile pads. The existing
observation call supplies image references before the policy image filter.
Publication on that path only rate-limits and replaces references; a background
thread copies and transfers them. Images never enter the motion FIFO, and the
display never reads cameras a second time. Producers must not mutate published
RGB arrays; the current Elite camera drivers publish replacement arrays.

Current Task shows the actual configured default prompt returned by the policy
server, its source, and runtime state. This metadata applies to the current
Elite requests without a prompt; a future request-level prompt needs its own
provenance. Old servers without task metadata show an unavailable instruction.
The UI labels and runtime events use English because the pinned Viewer does
not include CJK fonts. The actual instruction is preserved verbatim.
Sub Task and Current Action show `Not connected`. Task Timeline records actual runtime
events. The final reset is not a new episode, and episode completion does not
mean the task succeeded. Dry-run state explicitly says no robot action was sent.

The image timestamp is the observation sampling time, not exposure time.
A stalled observation stream clears its images after `image_stale_after` seconds;
it cannot detect a device repeatedly returning an old cache. Images follow the
observation cadence while SDK state continues on its independent sampler.

A dead or overloaded display does not block inference. Shutdown is bounded and
can terminate the display process. Output flush timeouts stop the display,
including its recording, rather than letting a failed sink grow a send backlog.
Inspect `session.errors` and display statistics; forced shutdown may leave an
incomplete RRD.

Display imports, mesh loading and viewer startup finish in the child process
**before connecting the robot**. The SDK sampler starts after connection.

Measured TacCap fingers use the Elite SDK snapshot reader independently of
whether station collision checking is enabled.

`VisualizationSession.set_enabled(False)` closes active subscriptions and the
display, and waits up to two seconds for the SDK sampler. If an SDK read remains
blocked, shutdown reports the error after closing the display, and re-enabling
is refused until that sampler exits. It never disconnects the robot.

Re-enabling starts a new recording and waits for the next chunk. Hot toggling
while PacedBroker still holds unsubmitted actions is not a complete reference
resynchronization: the old cached prefix can be absent from the new display.
For accurate full reference coverage, enable before starting the runtime and
keep it enabled through the episode.

Blue is remaining TCP reference; green is measured TCP history and the current
point. Robot link poses come from measured SDK joints and URDF forward
kinematics. Normal chunking displays only `action_horizon` usable rows. RTC
displays the effective queue **after** existing delay truncation and blending;
`action_horizon` remains a non-RTC setting. Actions already transferred to
PacedBroker are retained until successfully submitted to the environment while
the stream stays enabled.
Submission progress never serves as evidence of physical arrival. Dry-run
reference consumption still leaves the green path driven only by real state.

## Public subscriptions and robot-specific adapters

Optional code lives in `xense_client.action_stream`,
`xense_client.visualization` and `xense_client.visualization_brokers`; it has
no Elite names or 20D assumptions:

| Interface | Meaning |
|---|---|
| `session.status()` / `has_new_chunk(since_sequence)` | Enabled state and monotonically increasing chunk arrival sequence |
| `subscribe_chunks()` | Full copied chunk, usable length and host monotonic receipt time |
| `subscribe_plans()` | Complete remaining reference, stable action IDs and episode generation |
| `subscribe_robot_state()` | SDK state snapshots, sample identity and availability |
| `set_enabled(bool)` | Manage visualization resources; see blocked reads and hot-toggle limits above |

Subscriptions are bounded queues (`get(timeout=...)` or `latest()`), not user
callbacks on the inference thread. Full plan snapshots repair dropped display
updates. Arrays are immutable copies. The observed brokers emit IDs only when
a stream is enabled; the environment wrapper removes the display tag before
passing the action to the original environment.

The shared `ActionChunkBroker`, `RTCActionChunkBroker`, `ActionQueue`,
`PacedBroker` and runtime implementations are unchanged. Only this Elite entry
point, when visualization is enabled, selects `ObservedActionChunkBroker` or
`ObservedRTCActionChunkBroker`. They call the original broker/queue methods and
observe the actual results; scheduling, warmup, cropping and blending are not
reimplemented. Other robot examples continue selecting the shared classes.
Install the viewer explicitly from the submodule when needed. The parent
project and xense-client dependency declarations remain unchanged.

Observation still adds copying and lock acquisition on the enabled action path;
the SDK sampler also shares cache locks with robot observations. A separate
display process does not make CPU and memory-bandwidth costs zero. Measure observation/action timing with
visualization on/off in the target hardware environment.

To add another robot, implement `RobotVisualizationAdapter.describe()`,
`decode_actions()` and `read_state()` in that robot's example. Elite's private
implementation is `visualization.py`: it owns the 20D slicing, assembly and
tool frame alignment, RTSI joint/TCP reads and measured gripper calibration.
SDK TCP feedback is base-frame XYZ/RPY, converted once into the assembly world
frame. Controller angles map directly to the URDF without initial offsets.
Missing/stale SDK feedback hides the current green point and breaks the next
history segment; it never substitutes a policy target or zero joints.

## Not included (yet)

Pico4 intervention, LeRobot recording and the obs subscriber stream from the
BiFlexiv example are not ported.

### Developing the two repositories together

The integration requires XMV 0.6.0 with `LiveDisplayProcess`. During development,
install the matching sibling checkout into the robot-client environment:

```bash
python -m pip install -e /path/to/xense-motion-viewer'[elite,rerun]'
```

The `third_party/xense-motion-viewer` submodule pins the tested XMV 0.6.0
revision. Keep this pin synchronized when changing the display API. This
feature does not require a LeRobot source change.
