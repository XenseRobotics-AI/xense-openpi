# Resume and intervention-only recording

Use `recording_control` for the combined recording modes.

## First run: record only Pico4 takeover frames

```bash
mamba run -n lerobot-xense python -m \
  examples.bi_flexiv_rizon4_rt_record.recording_control \
  --host 192.168.200.148 \
  --port 8000 \
  --bi-mount-type diagonal-02 \
  --record-intervention \
  --record-repo-id Xense/my_dataset_expand \
  --task "insert optical module" \
  --num-episodes 3 \
  --keyboard-episode-control
```

`--record-intervention` automatically enables Pico4 intervention. Policy
inference frames are skipped. Each contiguous takeover interval is saved as a
LeRobot episode in the selected dataset.

## Later run: append to the same dataset

Run the same command with `--record-resume`:

```bash
mamba run -n lerobot-xense python -m \
  examples.bi_flexiv_rizon4_rt_record.recording_control \
  --host 192.168.200.148 \
  --port 8000 \
  --bi-mount-type diagonal-02 \
  --record-intervention \
  --record-resume \
  --record-repo-id Xense/my_dataset_expand \
  --task "insert optical module" \
  --num-episodes 3 \
  --keyboard-episode-control
```

The recorder loads the local dataset at `--record-root` or, by default,
`~/.cache/huggingface/lerobot/<record-repo-id>`, preserves its existing
episodes, and appends new episode indices after them. If no dataset exists at
that location, resume mode creates it.

`--record` and `--record-intervention` remain mutually exclusive. Use
`--record --record-resume` if full policy plus intervention episodes should be
appended instead.

