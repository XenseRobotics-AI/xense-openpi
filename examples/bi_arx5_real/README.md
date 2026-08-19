# Run BiARX5 (Real Robot)

This example demonstrates how to run with a real BiARX5 robot using the [lerobot BiARX5 implementation](https://github.com/huggingface/lerobot). The implementation is based on direct ARX5 SDK communication (no ROS required).

## Prerequisites

This implementation uses the lerobot BiARX5 robot integration located at `~/lerobot-ARX5`.

1. Ensure your BiARX5 hardware is properly set up with CAN bus interfaces
2. Configure CAN interfaces (typically `can1` and `can3` for dual arms)
3. Your lerobot environment should already have all necessary dependencies

### CAN Bus Setup

```bash
# Set up CAN interfaces (run as root or with sudo)
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can3 type can bitrate 1000000
sudo ip link set up can1
sudo ip link set up can3

# Or use the provided script from lerobot-ARX5
cd ~/lerobot-ARX5
sudo ./set_can.sh
```

## Usage (No Docker Required)

### Terminal 1: Start OpenPI Policy Server

```bash
cd /home/ubuntu/openpi
python scripts/serve_policy.py --env BI_ARX5 --default_prompt='pick and place cube'
```

### Terminal 2: Run BiARX5 Robot Client

```bash
cd /home/ubuntu/openpi

# Install minimal dependencies (if needed)
pip install -r examples/bi_arx5_real/requirements.txt
pip install -e packages/xense-client

# Run the robot client
python -m examples.bi_arx5_real.main \
    --left_arm_port=can1 \
    --right_arm_port=can3 \
    --action_horizon=25 \
    --log_level=INFO \
    --use_multithreading=true
```

## Synchronized Recording

Record a new LeRobot-format dataset while running inference (raw 640×480 images,
absolute joint actions — same format as the training data).

```bash
python -m examples.bi_arx5_real.main \
    --left_arm_port=can1 \
    --right_arm_port=can3 \
    --record \
    --record_repo_id Xense/my_new_dataset \
    --task "pick and place cube"
```

The dataset is saved locally to `~/.cache/huggingface/lerobot/<repo_id>` by default.
Use `--record_root /path/to/dir` to override the save location.

---

## Converting Recorded Dataset to MCAP

[MCAP](https://mcap.dev/) files can be opened in [Foxglove Studio](https://foxglove.dev/)
for visual inspection of observations, actions, and camera streams.

> Run in the `lerobot-xense` environment where `lerobot2mcap` is installed.

### Convert all episodes

```bash
mamba run -n lerobot-xense lerobot2mcap convert \
    ~/.cache/huggingface/lerobot/Xense/my_new_dataset \
    -o ~/mcap_output/my_new_dataset
```

### Convert specific episodes

```bash
mamba run -n lerobot-xense lerobot2mcap convert \
    ~/.cache/huggingface/lerobot/Xense/my_new_dataset \
    -o ~/mcap_output/my_new_dataset \
    --episodes 0 1 2
```

### Parallel conversion (faster for large datasets)

```bash
mamba run -n lerobot-xense lerobot2mcap convert \
    ~/.cache/huggingface/lerobot/Xense/my_new_dataset \
    -o ~/mcap_output/my_new_dataset \
    --jobs 4
```

Each episode produces a separate `.mcap` file under the output directory.
Open any `.mcap` file directly in Foxglove Studio to inspect it.

---

## Test the Environment

Before running with real hardware, test the environment:

```bash
cd /home/ubuntu/openpi
python examples/bi_arx5_real/test_env.py
```

This will:

1. Test environment creation without hardware
2. Test OpenPI integration compatibility
3. Optionally test real hardware connection (if you choose)

## Configuration Options

```bash
python -m examples.bi_arx5_real.main \
    --left_arm_port=can1 \
    --right_arm_port=can3 \
    --action_horizon=25 \
    --log_level=INFO \
    --use_multithreading=true \
    --max_episode_steps=1000 \
    --host=0.0.0.0 \
    --port=8000
```

## Robot Specifications

- **Model**: ARX5 (X5 variant)
- **DOF**: 6 joints per arm + 1 gripper per arm (14 total DOF)
- **Control**: Direct CAN bus communication via ARX5 SDK
- **Frequency**: 100Hz control loop (controller_dt=0.01)
- **Communication**: No ROS required

## Action Space

The action space consists of 14 dimensions:
- Left arm: 6 joint positions + 1 gripper position
- Right arm: 6 joint positions + 1 gripper position

Action format: `[left_j1, left_j2, ..., left_j6, left_gripper, right_j1, right_j2, ..., right_j6, right_gripper]`

## Observation Space

The observation space includes:
- `state`: 14-dimensional joint position vector
- `images`: Dictionary with camera images
  - `cam_high`: Head camera (224x224x3)
  - `cam_left_wrist`: Left wrist camera (224x224x3)
  - `cam_right_wrist`: Right wrist camera (224x224x3)

## Camera Configuration

The system expects three cameras configured in your lerobot BiARX5Config:

```python
cameras = {
    "head": RealSenseCameraConfig("230322271365", fps=30, width=640, height=480),
    "left_wrist": RealSenseCameraConfig("230422271416", fps=30, width=640, height=480),
    "right_wrist": RealSenseCameraConfig("230322274234", fps=30, width=640, height=480),
}
```

## Safety Features

- Gravity compensation mode during operation
- Smooth trajectory interpolation for reset/home movements
- Parallel arm control with timeout protection
- Automatic error handling and recovery

## Troubleshooting

### CAN Bus Issues
```bash
# Check CAN interfaces
ip link show can1
ip link show can3

# Monitor CAN traffic
candump can1
candump can3
```

### Robot Connection Issues
```bash
# Test lerobot BiARX5 directly
cd ~/lerobot-ARX5
python test_bi_arx5_lerobot.py

# Test OpenPI integration
cd /home/ubuntu/openpi
python examples/bi_arx5_real/test_env.py
```

### Permission Issues
```bash
# Add user to dialout group for device access
sudo usermod -a -G dialout $USER
# Logout and login again
```

## Integration with Trained Models

This environment is compatible with models trained using the lerobot framework on BiARX5 data:

1. **Use existing checkpoints**: Models trained on `Vertax/bi_arx5_pick_and_place_cube`
2. **Train custom models**: Use lerobot to record and train on your own tasks
3. **Load models**: Specify model path in the policy server

Example with custom model:
```bash
# Terminal 1: Start server with custom model
python scripts/serve_policy.py \
    --policy_path=/path/to/your/model \
    --default_prompt='your task description'

# Terminal 2: Run robot
python -m examples.bi_arx5_real.main
```

## File Structure

```
examples/bi_arx5_real/
├── main.py          # Main entry point
├── env.py           # OpenPI Environment adapter
├── real_env.py      # BiARX5 robot control (based on lerobot)
├── test_env.py      # Test script
├── requirements.txt # Minimal dependencies
└── README.md        # This file
```

The implementation leverages your existing lerobot BiARX5 setup, so no additional robot drivers or ROS components are needed.


test time best parameters: 
action_horizon: 50
controller_dt: 0.002
preview_time: 0.04
max_hz: 20
default_kp / 2
default_kd * 2
- low speed but stable control

test time best parameters: 
action_horizon: 50
controller_dt: 0.002
preview_time: 0.03
max_hz: 50
default_kp / 2
default_kd * 2
- high speed but low accuracy of control