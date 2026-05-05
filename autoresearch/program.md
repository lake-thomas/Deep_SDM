# Deep_SDM Autoresearch Program

You are an autonomous research agent improving Deep_SDM model training under a locked ecological benchmark. Your job is to make small, reviewable changes to `train.py`, run the fixed benchmark, compare against `results.tsv`, and keep only changes that improve the primary metric.

## Core idea

The autoresearch loop is:

1. Inspect `results.tsv` and the current `train.py` candidate.
2. Propose one small hypothesis-driven change.
3. Edit only `train.py`.
4. Run the fixed benchmark.
5. Compare the new row in `results.tsv` against the best prior result.
6. Keep the change if it improves the primary metric; otherwise revert or try a narrower follow-up.
7. Record a brief note describing what changed and why.

## Files and edit permissions

### Locked files: do not edit

- `prepare.py`
- dataset CSVs
- NAIP chips
- topographic chips
- normalization statistics
- split/fold assignments
- evaluation metric definitions
- `results.tsv`, except by running the benchmark

### Mutable file: edit only this

- `train.py`

The benchmark harness in `prepare.py` is the source of truth for data loading, feature order, train/validation splits, spatial CV folds, validation metrics, and result logging.

## Supported model families

Deep_SDM currently supports seven predictor-family combinations:

- `image_only`: NAIP imagery only
- `tabular_only`: WorldClim + GHM + lat/lon + optional topographic scalar summaries
- `topo_only`: topographic image chips only
- `image_tabular`: NAIP imagery + tabular predictors
- `topo_tabular`: topographic chips + tabular predictors
- `image_topo`: NAIP imagery + topographic chips
- `image_topo_tabular`: NAIP imagery + topographic chips + tabular predictors

## Primary objective

Optimize the benchmark's configured primary metric. The recommended default is:

```text
mean_val_mcc
```

MCC is preferred over raw accuracy because it gives a more balanced view of false positives and false negatives in binary species distribution modeling.

Tie-breakers, in order:

1. higher `mean_val_auc`
2. higher `mean_val_sensitivity`
3. higher `mean_val_specificity`
4. lower `mean_val_loss`
5. simpler model / fewer new assumptions

Never use the test split to guide edits. Test results are for final human review only.

## Allowed research changes

Good first changes include:

- `model_type`
- learning rate
- weight decay
- optimizer choice: Adam, AdamW, SGD, and others
- scheduler choice: none, ReduceLROnPlateau, cosine, step
- dropout
- hidden dimension
- image/topo/tabular branch feature dimensions
- whether ImageNet pretraining is used for the NAIP branch
- fusion head depth or normalization, if implemented inside `train.py`
- small custom encoder variants, if implemented cleanly inside `train.py`
- seed, but only last and when explicitly testing robustness after a promising configuration

## Forbidden changes

Do not change anything that alters the ecological benchmark:

- presence/background labels
- train/validation/test split
- spatial CV fold assignment
- background sampling method
- occurrence filtering
- normalization statistics
- tabular feature leakage exclusions
- evaluation metrics
- threshold used for fixed-threshold metrics unless the human explicitly asks
- paths to a different dataset unless the benchmark config is intentionally changed by the human

Do not add leakage features such as:

- `nearest_presence_km`
- `background_sampling_rule`
- `filename`
- `url`
- `block_id`
- `fold`
- `cv_round`
- `topo_valid_frac`
- `topo_normalized`

## Experiment strategy

Use phases.

### Phase 1: baseline sweep

Run the seven model families with conservative defaults. Establish which predictor family is strongest on the scout benchmark.

### Phase 2: hyperparameter tuning

For the best two or three model families, tune:

- learning rate: `1e-5`, `3e-5`, `1e-4`, `3e-4`
- weight decay: `0`, `1e-5`, `1e-4`, `1e-3`
- dropout: `0.15`, `0.25`, `0.35`, `0.50`
- hidden dimensions: `128`, `256`, `512`
- branch dimensions: `64`, `128`, `256`

Change one thing at a time unless there is a clear reason to test a coupled change.

### Phase 3: multimodal fusion

For `image_topo_tabular`, test whether branch dimensions should be balanced:

```text
NAIP branch -> 128 or 256 features
topo branch -> 128 or 256 features
tabular branch -> 128 or 256 features
```

Prefer stable, modest-capacity models over very large models unless validation metrics clearly improve.

### Phase 4: robustness

After finding a strong configuration, test 3 seeds and multiple spatial folds. Do not declare a result improved based on one lucky seed.

## How to run

From the Deep_SDM repository root:

```bash
python sdm_autoresearch/prepare.py --write-example-config sdm_autoresearch/configs/scout_template.json
```

Edit the config paths once. Then run one candidate:

```bash
python sdm_autoresearch/train.py --config sdm_autoresearch/configs/test_species_mini_scout.json
```

Inspect:

```text
autoresearch_runs/results.tsv
```

## Reporting format after each trial

After running a trial, summarize:

```text
Hypothesis:
Change:
Result:
Primary metric change:
Decision: keep/revert/follow-up
Next experiment:
```

## Scientific guardrails

More predictors do not guarantee better SDMs. Multimodal models can overfit or degrade if one branch is noisy or poorly regularized. Treat improvements as provisional until they hold across spatial CV folds and, ideally, independent validation data.
