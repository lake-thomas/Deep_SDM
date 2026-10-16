# Training

Training reads a dataset built as described in [Data](data.md) and fits one model. All settings live in a JSON configuration file:

```bash
python main.py --config my_config.json
```

Results go to `output_dir/experiment/`. Training refuses to write into a folder that already holds results, so earlier runs are never overwritten by accident. Pass `--allow-existing-output` to override this.

## Model types

Every model has up to three input branches whose outputs are combined by a small classifier:

- an **image branch**, a ResNet-18 network that reads the 4-band NAIP chip. With `pretrained_image` it starts from ImageNet weights; the near-infrared band starts from the average of the red, green and blue filters;
- a **terrain branch**, a small convolutional network that reads the terrain chip;
- a **tabular branch**, a small fully connected network that reads the climate, land use, location and terrain summary variables.

`model_type` chooses which branches to use:

| `model_type` | Inputs |
|---|---|
| `image_only` | Image chip |
| `tabular_only` | Tabular variables |
| `topo_only` | Terrain chip |
| `image_tabular` | Image chip and tabular variables (recommended) |
| `topo_tabular` | Terrain chip and tabular variables |
| `image_topo` | Image chip and terrain chip |
| `image_topo_tabular` | All three (also called `full`) |

The recommended `image_tabular` model includes terrain as the ten summary variables in the tabular inputs (`include_topo_scalars: true`), so it does not need the terrain chips.

## Recommended configuration

[`examples/train_config.json`](../examples/train_config.json) holds the recommended starting point:

- **Model:** `image_tabular` with a pretrained image network, a hidden size of 192 and dropout of 0.25.
- **Optimization:** a learning rate of 0.00001 and batches of 128. The learning rate is halved after 5 epochs without improvement in validation loss, and training stops after 10 such epochs or 100 epochs in total. The best model is kept.
- **Augmentation:** random 90-degree rotations and flips of the image chip (`dihedral`), and mild random changes to brightness, contrast and individual bands (`radiometric_augmentation`). A check rejects color changes that would shift the chip's average vegetation index by more than a set amount.
- **Labels:** light label smoothing (0.05), which softens the 0/1 targets and helps with presence/background data, where some background points are unrecorded presences.
- **Same-ground consistency** with a weight of 1.0. See [Same-ground consistency](same_ground_consistency.md).

A few settings are worth tuning for each species with spatial cross-validation: dropout (0.25 to 0.35), label smoothing (0 to 0.05), the strength of the radiometric augmentation, and whether to weight presences more heavily (`loss_strategy: "pos_weight_bce"`).

## Configuration reference

Required keys: `csv_path`, `image_dir` (the folder that chip paths are relative to), `output_dir`, `batch_size`, `epochs` and `learning_rate`.

| Key | Default | Purpose |
|---|---|---|
| `experiment` | `host_sdm_<model_type>` | Name of the output folder |
| `model_type` | `full` | Which input branches to use |
| `env_features` | `auto` | Tabular variables. `auto` uses every standard variable present in the CSV. |
| `include_topo_scalars` | `true` | Add the terrain summaries to the tabular variables when `env_features` is `auto` |
| `pretrained_image` | `true` | Start the image network from ImageNet weights |
| `hidden_dim`, `env_feature_dim`, `topo_feature_dim` | 256, 128, 128 | Layer sizes |
| `dropout` | 0.25 | Dropout rate in the tabular, terrain and classifier layers |
| `spatial_augmentation` | `legacy_rotation` | `dihedral`, `legacy_rotation` (random rotation up to `rotation_degrees`, which defaults to 0) or `none` |
| `radiometric_augmentation` | `false` | Random brightness, contrast and band changes (a settings object, see the example) |
| `loss_strategy` | `ordinary_bce` | `ordinary_bce`, `pos_weight_bce` (weights presences by the class ratio) or `balanced_batches` |
| `label_smoothing` | 0 | Label smoothing for the training loss |
| `same_ground_consistency` | off | See [Same-ground consistency](same_ground_consistency.md) |
| `image_harmonization` | off | Optional band standardization for each chip, see the same guide |
| `lr_patience`, `es_patience` | 5, 10 | Epochs without improvement before halving the learning rate and before stopping |
| `use_fp16` | `true` | Mixed precision on a GPU |
| `num_workers` | 0 | Data loading processes |
| `postfit_calibration` | `none` | `platt` fits a Platt scaling calibrator on the validation set |
| `seed`, `deterministic_training` | 42, `true` | Reproducibility |
| `use_wandb` | `false` | Log to Weights & Biases (`wandb_project`, `wandb_entity`) |

Validation loss always uses the unweighted, unsmoothed loss, so runs with different loss settings can be compared fairly.

**Listing variables explicitly.** With `env_features: "auto"`, training uses the standard variables that are present in the CSV, while inference assumes all of them. If your dataset leaves any out, list the variables you use in `env_features` so training and inference agree.

## Standardization statistics

Add the dataset's standardization statistics to the configuration so they are saved with the model and reused at inference (see [Data](data.md#standardization-statistics)). Training ignores these keys. Each entry gives a mean and a standard deviation:

```json
"worldclim_stats": {
  "wc2.1_30s_bio_1": {"mean": 10.97, "std": 5.09},
  "wc2.1_30s_bio_2": {"mean": 13.65, "std": 2.17}
},
"lat_lon_stats": {
  "lat": {"mean": 39.34, "std": 4.24},
  "lon": {"mean": -90.41, "std": 15.36}
},
"topo_normalization_stats": {
  "elevation": {"mean": 726.3, "std": 726.1}
}
```

Include all 19 bioclimatic variables and all four terrain layers (`elevation`, `slope`, `northness`, `eastness`). The numbers above only show the format.

## Outputs

| File | Contents |
|---|---|
| `checkpoints/best.pt` | The best model, with its settings and feature list |
| `config.input.json` | The configuration used. Pass it to inference. |
| `training_history.csv` | Loss and metrics for each epoch |
| `training_design.json` | Loss, sampling and augmentation settings and a checksum of the dataset |
| `runtime_provenance.json` | Code version, host, platform and start time |
| `validation_threshold.json` | Decision threshold that maximizes the Matthews correlation coefficient on the validation set |
| `prediction_results.csv`, `model_summary_metrics.csv` | Test set predictions and metrics |
| ROC curve, precision and recall curve, error maps | Test set figures |

Models trained on presence and background points produce a relative suitability score. Use it to compare places. It is not the probability that the species is present.

## Spatial cross-validation

The builder writes five fold files based on 200 km spatial blocks. To estimate how a configuration transfers to unseen regions, train once per fold by pointing `csv_path` at each fold file and compare test metrics across folds.
