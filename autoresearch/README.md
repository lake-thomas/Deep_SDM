# Deep_SDM Autoresearch Harness

This folder adapts the `autoresearch` idea to Deep_SDM.

- `prepare.py` is the locked benchmark/evaluation harness. Do not edit during agent runs.
- `train.py` is the mutable research file. Agents edit this file only.
- `program.md` is the instruction file for Codex/Claude/other coding agents.
- `configs/test_species_mini_scout.json` is a ready-to-edit scout config using the Test Species Mini path layout.

## From the Deep_SDM repo root

```bash
python sdm_autoresearch/prepare.py --config sdm_autoresearch/configs/test_species_mini_scout.json
python sdm_autoresearch/train.py --config sdm_autoresearch/configs/test_species_mini_scout.json
```

`prepare.py` also accepts a regular Deep_SDM training config with top-level `csv_path` and `image_dir`.
It will coerce that into a benchmark config internally.

## Output

Results are appended to:

```text
<output_dir>/results.tsv
```

Each candidate also gets a run directory containing:

```text
benchmark_config.json
search_space.json
fold_metrics.csv
trial_result.json
```
