# Deep_SDM

Deep_SDM is a deep learning framework for species distribution modeling that combines spatial and topographic context with structured predictors in a unified workflow.

## Tiled-inference overlap blending

`inference_multimodal_production.py` retains flat overlap averaging by default.
For the August production path, use the validated center-weighted window:

```bash
--blend-window hann --blend-edge-floor 0.05
```

The positive edge floor prevents Hann weighting from creating new zero-support
pixels within the existing chip coverage. Prediction and uncertainty rasters
receive the same normalized window; the count raster remains the unweighted
integer number of contributing chips. The applied mode is recorded in the
inference summary and output raster tags.
