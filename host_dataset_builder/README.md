# Dataset builder

This package builds Deep SDM training datasets from occurrence records, NAIP imagery and environmental rasters. For a walkthrough with the recommended settings, see [docs/data.md](../docs/data.md).

## Entry points

Run these from the repository root. Each one lists its options with `--help`.

| Command | Purpose |
|---|---|
| `python -m host_dataset_builder` | Build a complete dataset in one step |
| `python -m host_dataset_builder.candidate_manifests` | Sample presence and background points, check their environmental values and write nested manifests for several background sizes, without extracting imagery |
| `python -m host_dataset_builder.materialize_candidates` | Extract the chips for a reviewed manifest package |

Use `--occurrence-dir` instead of `--occurrence-file` to build datasets for a folder of species files.

## Modules

| Module | Role |
|---|---|
| `io_utils.py` | Load occurrence files and index NAIP tiles |
| `background.py` | Deduplicate presences and sample background points |
| `naip.py` | Match points to NAIP acquisitions and extract image chips |
| `env.py` | Sample and standardize climate and land use variables |
| `topo.py` | Extract and standardize terrain chips and summaries |
| `splits.py` | Training, validation and test splits, spatial blocks and folds |
| `qa.py` | Quality checks on the finished dataset |
| `dataset_builders.py` | Write the dataset CSV |
| `pipeline.py`, `cli.py` | Build in one step, and its command options |
| `candidate_manifests.py`, `candidate_environment_qa.py`, `materialize_candidates.py` | Build in two steps |

## Options for checking a build

- `--plan-only` checks the occurrence file and prints the planned outputs without reading rasters or writing files.
- `--background-domain-plan-only` reads the tile index, applies any `--background-allowed-states` restriction and lists the acquisition groups available for background sampling.
- `--max-presences-after-thinning` caps the number of presences, for quick trial builds.
- The builder refuses to overwrite an existing dataset. Pass `--allow-existing-output` to write into a folder that already holds results.

## Tracking image sources

With `--naip-source-lock-mode strict`, every background point carries the NAIP tile, state, acquisition date and acquisition group it was sampled from, and its chip is read only from that acquisition. A chip that crosses a tile edge may combine several tiles only when they share the same state and acquisition date. The dataset records the intended and the actual source of every chip, how many tiles it combines, its fraction of valid pixels and whether the two sources match.

With `--background-sampling-design domain_balanced`, background sampling first places a minimum number of points in every acquisition group (state, year and month by default) and then spreads the remaining points by area. `--background-domain-columns` changes how groups are defined.

## Quality checks

With `--manifest-qa-mode strict`, the builder writes a `qa/` folder and rejects the dataset if it finds missing required fields, duplicates, conflicting labels, leakage between splits or spatial folds, missing files, non-finite predictor values, chips that mix acquisitions, or chips whose actual source differs from the intended one. The number of points per acquisition group and the share of presences are reported for review without rejecting the dataset.

## Occurrence files

Latitude and longitude columns are detected automatically (`decimalLatitude`, `latitude` or `lat`, and the matching longitude names), as are common GBIF and iNaturalist coordinate uncertainty fields. Filter records by coordinate uncertainty before building, for example with `--max-coordinate-uncertainty-m 128`.
