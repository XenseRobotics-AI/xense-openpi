# Keyboard-gated episode control

The original `main.py` behavior is unchanged. To manually gate episode
transitions, run the keyboard entry point and enable `--keyboard_episode_control`:

```bash
mamba run -n lerobot-xense python -m \
  examples.bi_flexiv_rizon4_rt_record.keyboard_main \
  --host 10.142.1.1 --port 8000 \
  --record \
  --record_repo_id Xense/my_dataset \
  --task "insert optical module" \
  --num_episodes 10 \
  --keyboard_episode_control
```

The first episode starts normally. During inference:

1. Press **Right Arrow** once to end the current episode. The recorder saves
   the episode, the robot is reset to its initial position, and no policy
   actions are sent while reset mode is active.
2. After the reset completes, press **Right Arrow** again to start the next
   inference episode.

The terminal running the command must have focus. This mode uses the
synchronous runtime, so keep `--action_hz 0` (the default); it is not compatible
with `--action_hz > 0`.

Without `--keyboard_episode_control`, `keyboard_main` delegates to the
original entry point and retains the existing automatic episode behavior.

