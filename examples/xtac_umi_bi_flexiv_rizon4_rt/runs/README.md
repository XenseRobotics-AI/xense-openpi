# Run configs — XTac-UMI on bimanual Flexiv Rizon4 RT

One file per launch; `--args.run <name>` presets the CLI flags:

```bash
python -m examples.xtac_umi_bi_flexiv_rizon4_rt.main --args.run dry-run --args.host 192.168.142.220
```

Keys are the CLI field names (see `main.py --help`), precedence is
`defaults < run YAML < CLI flags`.

The bench comes from `robot_recipe:`, which resolves against
[`../../bi_flexiv_rizon4_rt/recipes/`](../../bi_flexiv_rizon4_rt/recipes/) —
this example ships no recipes of its own so the benches stay defined once for
the repo. `forward-04` is the taccap-gripper bench.

Full documentation of the mechanism, including run-vs-recipe:
[`../../bi_flexiv_rizon4_rt/runs/README.md`](../../bi_flexiv_rizon4_rt/runs/README.md).
