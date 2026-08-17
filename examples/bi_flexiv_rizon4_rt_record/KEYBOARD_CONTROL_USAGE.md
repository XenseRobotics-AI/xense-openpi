# Right Arrow episode control

Use the flat CLI alias:

```bash
mamba run -n lerobot-xense python -m \
  examples.bi_flexiv_rizon4_rt_record.keyboard_control \
  --host 10.142.1.1 --port 8000 \
  --record \
  --record-repo-id Xense/my_dataset \
  --task "insert optical module" \
  --num-episodes 10 \
  --keyboard-episode-control
```

The first episode starts immediately. Press Right Arrow once to finish it and
enter reset mode. The recorder closes the episode, the robot returns to its
initial position, and no inference/action is performed while waiting. Press
Right Arrow a second time after reset completes to start the next episode.

Leave `--action-hz` at its default `0`; keyboard gating is implemented for the
synchronous runtime. Without `--keyboard-episode-control`, the alias delegates
to the existing automatic episode behavior.

