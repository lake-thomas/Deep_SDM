# Same-ground consistency

Same-ground consistency training keeps a model's predictions stable when the same place appears in two different aerial images. Use it for any model with an image branch that will map large areas.

## The problem

NAIP imagery is flown state by state, one file at a time, so neighboring image files can differ in date, sensor and processing. An image network can pick up on these differences and learn cues tied to a particular file instead of the landscape. The resulting maps show a blocky pattern whose edges follow the image files, and the same place gets different predictions depending on which file is read. Adjusting the colors of each chip does not fix this, because the differences between files go beyond overall brightness and contrast.

## The idea

Neighboring NAIP files overlap along their edges, and areas near state borders often appear in two state collections. These overlaps give pairs of chips that cover exactly the same ground in two different acquisitions. During training the model sees these pairs and pays a penalty whenever it scores the two chips of a pair differently. Because both chips show the same place, the only way to avoid the penalty is to ignore what is specific to each image file.

## How it works

Each training batch combines the usual labeled chips with a batch of same-ground pairs (64 by default). The training loss is

```
loss = supervised loss + weight × mean of (score_A − score_B)²
```

where `score_A` and `score_B` are the model's raw scores for the two chips of a pair, before conversion to a 0 to 1 suitability value. Some details matter:

- Both chips of a pair share one set of tabular values (climate, land use, location and terrain), so only the image differs between them.
- The pair term is computed with the model in evaluation mode: dropout is off and batch normalization uses its stored statistics. The model is therefore made consistent in the same state it is in when it makes maps.
- Each pair receives the same random rotation and flip.
- Pairs are reused in a new random order once all of them have been seen.

The pair term needs no labels, so pairs can come from anywhere in the mapped region.

## Building pairs

A pair is two 4-band chips of the same size as the training chips, read from the same footprint in two different NAIP files. Good pairs follow these rules:

- **Overlap.** The two files must overlap by slightly more than one chip in both directions (at least 520 m for 512 m chips). Place the chip at a random position inside the overlap.
- **Complete pixels.** Keep a pair only if at least 99% of the pixels in both chips are valid.
- **Shared tabular values.** Compute the tabular variables once, at the center of the footprint, and use them for both chips. Apply the same standardization as the training dataset.
- **Evaluation pairs.** Set some pairs aside, in areas spatially separate from the training pairs, to measure consistency after training. For example, assign a pair to evaluation when (floor(longitude) + floor(latitude)) mod 4 equals 0.

About 25,000 to 35,000 training pairs and a few thousand evaluation pairs spread across the mapped region worked well.

## Pair file format

Save the pairs in one folder:

| File | Contents |
|---|---|
| `images_a.npy` | Unsigned 8-bit array of shape (N, 4, H, W): the first chip of each pair |
| `images_b.npy` | Same shape: the second chip of each pair |
| `env.npy` | Float array of shape (N, F): the tabular values shared by both chips, in the model's feature order |
| `keep.npy` | Boolean array of length N: pairs to use |
| `meta.csv` | One row per pair with a `split` column, `train` or `eval` |

`F` must equal the number of tabular variables the model uses. Training checks this and stops with an error if they differ.

## Configuration

```json
"model_type": "image_tabular",
"same_ground_consistency": {
  "enabled": true,
  "weight": 1.0,
  "pairs_dir": "pairs/my_species",
  "batch_size": 64,
  "num_workers": 4
}
```

A weight of 1.0 worked well. The pair term supports models with image and tabular inputs (`image_tabular`); include terrain through the tabular summaries. At the start of training, `main.py` prints the weight and the number of training pairs, and `training_design.json` records the settings.

## Checking the result

After training, score both chips of every evaluation pair and compute the share of pairs whose suitability values differ by more than 0.1. Then look at the map for patterns that follow image file edges.

In a tree-of-heaven model, this share was 12 to 14% without same-ground training and below 0.5% with it, and the blocky pattern disappeared from the map. Test AUC barely changed (0.976 to 0.980).

## Image harmonization

`image_harmonization` is an optional, separate setting. In `chip_standardize` mode it rescales each band of every chip to a fixed reference mean and standard deviation, during training, for same-ground pairs and at inference:

```json
"image_harmonization": {
  "mode": "chip_standardize",
  "ref_mean": [0.35, 0.38, 0.33, 0.45],
  "ref_std": [0.08, 0.07, 0.06, 0.10]
}
```

Compute `ref_mean` and `ref_std` from the training chips (for example, the average over chips of each band's mean and standard deviation over a few thousand chips, after dividing pixel values by 255). The values above only show the format.

Harmonization on its own did not reduce disagreement between same-ground pairs in the tree-of-heaven model, and adding it to same-ground training made little further difference. The recommended configuration leaves it off.
