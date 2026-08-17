# Run configs — single-arm Flexiv Rizon4 RT

One file per launch; `--args.run <name>` presets the CLI flags:

```bash
python -m examples.flexiv_rizon4_rt.main --args.run dry-run --args.host 192.168.2.215
```

Keys are the CLI field names (see `main.py --help`), precedence is
`defaults < run YAML < CLI flags`, and the bench still comes from a
[`../recipes/`](../recipes/) entry named by `robot_recipe:`.

Full documentation of the mechanism, including run-vs-recipe:
[`../../bi_flexiv_rizon4_rt/runs/README.md`](../../bi_flexiv_rizon4_rt/runs/README.md).
