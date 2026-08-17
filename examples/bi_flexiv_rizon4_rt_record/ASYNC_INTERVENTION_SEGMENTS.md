# Asynchronous intervention-segment recording

With `recording_control --record-intervention`, a runtime episode may contain
any number of Pico4 takeovers:

```text
policy -> intervention segment 1 -> policy -> intervention segment 2 -> policy
```

Each contiguous intervention segment is stored as a separate LeRobot episode.
Releasing the Pico4 grips only closes that recording segment; it does not end
the runtime episode, reset the robot, or stop policy inference.

Frames are copied into a FIFO queue from the control loop. Image writes, video
encoding, and `save_episode()` run on a dedicated recorder thread. Camera videos
are encoded sequentially to reduce CPU bursts. On process shutdown the queue is
drained before the dataset is finalized.

Policy frames are ignored in intervention-only mode. Add `--record-resume` to
append the newly saved segments to an existing local dataset.

