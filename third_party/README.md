# Optional Xense Motion Viewer integration

`xense-motion-viewer/` is a Git submodule of
<https://github.com/XenseRobotics-AI/xense-motion-viewer>. The parent repository
pins an exact commit, so every checkout uses the same reviewed viewer code.

Initialize it in an existing checkout:

```bash
git submodule update --init third_party/xense-motion-viewer
python -m pip install -e 'third_party/xense-motion-viewer[rerun]'
```

New checkouts can use `git clone --recurse-submodules`. Normal inference without
visualization does not require initializing or installing this submodule.

To adopt a later viewer revision, update the submodule to that reviewed commit
and include its gitlink in the xense-openpi change:

```bash
git -C third_party/xense-motion-viewer fetch origin
git -C third_party/xense-motion-viewer checkout <reviewed-commit>
git add third_party/xense-motion-viewer
```

Check the current pin with `git submodule status`. A leading `+` means the local
submodule checkout differs from the commit pinned by xense-openpi. Copy scripts
and file-hash manifests are no longer used to update the dependency.

The viewer pins Rerun 0.26.2 for the lerobot environment. See
`examples/bi_elite_cs66_rt/README.md` for the recipe switch and integration scope.
