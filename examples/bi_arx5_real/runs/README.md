# Run configs — BiARX5

One file per launch; `--args.run <name>` presets the CLI flags:

```bash
python -m examples.bi_arx5_real.main --args.run tactile --args.host 192.168.2.215
```

Keys are the CLI field names (see `main.py --help`), precedence is
`defaults < run YAML < CLI flags`. This example has no bench recipe layer, so
hardware knobs (`left_arm_port`, `right_arm_port`) live here too.

Full documentation of the mechanism:
[`../../bi_flexiv_rizon4_rt/runs/README.md`](../../bi_flexiv_rizon4_rt/runs/README.md).
