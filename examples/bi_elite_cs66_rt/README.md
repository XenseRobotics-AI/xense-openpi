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

Ctrl+C stops the runtime and disconnects the robot; a second Ctrl+C forces exit.

## Optional Xense Motion Viewer visualization

The display is off by default. It adds read-only chunk/progress subscriptions
and an independent sampler of the existing Elite SDK RTSI/gripper streams.
It does not read policy observations to obtain joints, open another robot
connection, call IK, or change action queue replacement/blending.

XMV pins Rerun `0.26.2`, within lerobot's supported range. Install in the existing
robot/inference environment from the xense-openpi repository root:

```bash
git submodule update --init third_party/xense-motion-viewer
python -m pip install -e 'third_party/xense-motion-viewer[rerun]'
```

Enable the top-level block in the robot recipe (beside `robot:`):

```yaml
visualization:
  enabled: true
  display_hz: 30
  spawn_viewer: true
  viewer_port: 9877
  save_path: null  # or recordings/elite-live.rrd
```

The renderer receives immutable data through a bounded queue on its own
thread; it has no robot object or SDK callbacks. A renderer startup failure is
logged and inference continues with visualization disabled.

`--args.visualize True` overrides the recipe for one launch; `--args.visualize False`
forces it off. The same `visualize: true/false` setting can be used in a run
YAML. When off, unused display settings are ignored; no subscriptions, state
sampler or renderer thread are created, and XMV/Rerun are not imported.
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
the SDK sampler also shares cache locks with robot observations. A renderer
thread does not make those costs zero. Measure observation/action timing with
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
