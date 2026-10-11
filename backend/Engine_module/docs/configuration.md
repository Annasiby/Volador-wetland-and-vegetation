# GIS Wetland Degradation & Depletion Engine — Configuration Reference

The GIS Wetland Engine is configured using a single authoritative JSON file (typically `configs/config.json`). This document defines the configuration schema, parameter scopes, and integration options.

---

## 1. Complete Configuration Schema

Here is a typical production configuration file with all necessary sections:

```json
{
    "model": {
        "path": "models/best_finetuned_model.onnx",
        "in_channels": 5,
        "num_classes": 4,
        "channel_order": ["ndvi", "gndvi", "ndre", "nir", "rededge"]
    },
    "preprocessing": {
        "mode": "inference",
        "band_scaling_factor": 10000.0,
        "require_crs": true,
        "epsilon": 1e-6,
        "min_denominator": 0.05,
        "min_band_value": 0.01,
        "gaussian_sigma": 1.0,
        "normalize": true
    },
    "ingestion": {
        "band_mapping": {
            "green": ["*_B03.tif", "*_B03_10m.tif", "*green.tif"],
            "red": ["*_B04.tif", "*_B04_10m.tif", "*red.tif"],
            "rededge": ["*_B05.tif", "*_B05_20m.tif", "*rededge.tif"],
            "nir": ["*_B08.tif", "*_B08_10m.tif", "*nir.tif"]
        }
    },
    "water": {
        "ndwi_threshold": 0.0,
        "epsilon": 1e-6
    },
    "change": {
        "residual_stress": {
            "enabled": true,
            "threshold": {
                "method": "percentile",
                "value": 95.0
            }
        }
    },
    "fusion": {
        "independent_hydrological_source_accepted": false
    },
    "wetland": {
        "source": "none"
    },
    "metrics": {
        "severity_weights": {
            "healthy": 0.0,
            "moderate": 0.35,
            "severe": 0.80,
            "background": 1.0
        }
    },
    "output": {
        "colormap": {
            "0": [128, 128, 128, 255],
            "1": [255, 0, 0, 255],
            "2": [255, 165, 0, 255],
            "3": [0, 170, 0, 255]
        }
    }
}
```

---

## 2. Parameter Detail Reference

### `model` Section
- `path` (String): Path to the trained U-Net model ONNX file. Overridden when running from CLI.
- `in_channels` (Integer): Number of tensor channels expected by the model input layer. (Must be 5).
- `num_classes` (Integer): Number of categorical classes output by U-Net argmax (Must be 4).
- `channel_order` (Array): Order of indices/bands built into the input tensor. (Must match `["ndvi", "gndvi", "ndre", "nir", "rededge"]`).

### `preprocessing` Section
- `mode` (String): Preprocessing target, either `"inference"` or `"training"`.
- `band_scaling_factor` (Float): Scaling divisor to convert raw integers (Sentinel-2 L2A DN) to reflectance values [0, 1]. Typically `10000.0`.
- `require_crs` (Boolean): If `true`, fails input validation if band files lack a coordinate reference system.
- `epsilon` (Float): Small offset value added to prevent divide-by-zero during index calculations.
- `min_denominator` (Float): Minimum denominator value for index equations. Denominators smaller than this are masked.
- `min_band_value` (Float): Low-end cutoff reflectance value to filter out shadow/negative noise.
- `gaussian_sigma` (Float): Standard deviation for Gaussian spatial smoothing applied during index computation.
- `normalize` (Boolean): Enables min-max normalization on the calculated index arrays.

### `ingestion.band_mapping` Section
Defines list of glob pattern strings to automatically map file names inside the scene directory to green, red, rededge, and nir spectral bands. The search is case-insensitive.

### `water` Section
- `ndwi_threshold` (Float): Threshold value to classify water pixels from the calculated NDWI array. Default is `0.0`. NDWI $\ge$ threshold designates water.
- `epsilon` (Float): Safety offset value for water index computation.

### `change.residual_stress` Section
- `enabled` (Boolean): Enables residual stress classification (integrating transitions and CVA outlier magnitudes).
- `threshold` (Object): Configuration for outlier detection.
  - `method` (String): Outlier calculation method, typically `"percentile"`.
  - `value` (Float): Percentile cutoff value (e.g. `95.0` designates pixels in the top 5% of CVA changes).

### `fusion` Section
- `independent_hydrological_source_accepted` (Boolean): Configures whether physical depletion can be assessed. If `true` and a valid wetland boundary is present, physical depletion assessment status is evaluated as `SUPPORTED_BY_INDEPENDENT_EVIDENCE`.

### `wetland` Section
- `source` (String): General type identifier for wetland boundaries. Defaults to `"none"`.

### `metrics.severity_weights` Section
Defines the weight multipliers applied to class areas when computing severity-weighted degradation metrics.
- `healthy`: typically `0.0`
- `moderate`: typically `0.35`
- `severe`: typically `0.80`
- `background`: typically `1.0`

### `output.colormap` Section
Color mappings applied when saving uint8 classification maps (`t1_health.tif`, `t2_health.tif`) as 8-bit paletted GeoTIFFs. Keys are class IDs (`0` to `3`) and values are lists representing RGBA values (`[R, G, B, A]`).
- `"0"` (Background): Grey `[128, 128, 128, 255]`
- `"1"` (Severe Stress): Red `[255, 0, 0, 255]`
- `"2"` (Moderate Stress): Orange `[255, 165, 0, 255]`
- `"3"` (Healthy): Green `[0, 170, 0, 255]`
