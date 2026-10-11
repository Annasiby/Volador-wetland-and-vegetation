# Phase 3 Implementation Plan - Two-Date Model Inference

This plan details the design and implementation steps for Phase 3 of the GIS Wetland Degradation & Depletion Engine. Phase 3 coordinates running the frozen ONNX model on two spatially aligned dates T1 (baseline) and T2 (comparison) and generates georeferenced health maps, logits, softmax probabilities, confidence maps, overlap masks, class statistics, and structured metadata.

## User Review Required

> [!IMPORTANT]
> - **Locked Model**: The ONNX model `models/best_finetuned_model.onnx` is frozen and will not be modified or retrained.
> - **Spatial Contract**: The T1 and T2 inputs must already be spatially aligned. If they are not aligned (i.e. if CRS, transform, resolution, width, height, or extent do not match exactly), the engine will fail immediately.
> - **NoData Handling**: Invalid pixels (as defined by NoData, NaNs, or Infinite values) will be explicitly set to a NoData value of `255` in the output integer health GeoTIFF maps, and will *not* be counted as class `0` (Background/Bare Soil) in statistics.

## Open Questions

None. The prompt details the entire API contract and requirements precisely.

---

## Proposed Changes

### Component: Inference & Probabilities
#### [MODIFY] [probabilities.py](file:///c:/Users/HP/OneDrive/Desktop/wetLand_Depreciation/src/gis_engine/inference/probabilities.py)
- Implement `calculate_softmax(logits: np.ndarray) -> np.ndarray`: Computes softmax probabilities along the class axis and returns a array of shape `(4, H, W)`.
- Implement `calculate_confidence(probabilities: np.ndarray) -> np.ndarray`: Computes the maximum probability over classes at each pixel, returning shape `(H, W)`.

---

### Component: Core Structures & Orchestration
#### [NEW] [results.py](file:///c:/Users/HP/OneDrive/Desktop/wetLand_Depreciation/src/gis_engine/core/results.py)
- Define `TwoDateInferenceResult` dataclass to store the results of two-date inference, including health maps, logits, probabilities, confidence, validity/overlap masks, reference grid, metadata, and class statistics.

#### [NEW] [pair.py](file:///c:/Users/HP/OneDrive/Desktop/wetLand_Depreciation/src/gis_engine/core/pair.py)
- Implement `analyze_pair(t1_scene: Path, t2_scene: Path, model_path: Path, output_dir: Path, config: Dict[str, Any]) -> TwoDateInferenceResult`:
  1. Initialize `TemporalAligner` to load T1/T2 bands and representative metadata.
  2. Verify that T1 and T2 share the exact same spatial grid (CRS, transform, width, height, resolution, bounds). Raise `ValueError` if mismatch.
  3. Execute `SingleDateInferenceEngine` for T1 and T2 to perform preprocessing, tensor compilation, and model prediction.
  4. Preserve raw logits for both dates.
  5. Compute class maps from logits using `argmax`.
  6. Calculate softmax probabilities and maximum confidence map for both dates.
  7. Compute validity masks for T1/T2 bands and create the combined overlap mask.
  8. Propagate the valid masks onto the class maps by marking invalid pixels with a `255` NoData value.
  9. Calculate class distribution statistics (counts and percentages) for both date-valid masks and the common overlap mask.
  10. Write outputs:
      - `T1/health.tif` and `T2/health.tif` (masked class maps georeferenced with colormap).
      - `T1/logits.npy` and `T2/logits.npy`.
      - `T1/probabilities.npy` and `T2/probabilities.npy`.
      - `T1/confidence.tif` and `T2/confidence.tif`.
      - `T1/metadata.json` and `T2/metadata.json`.
      - `overlap_mask.tif`.
      - `pair_metadata.json` (root metadata).
  11. Return the structured `TwoDateInferenceResult`.

---

### Component: Data Output
#### [MODIFY] [geotiff.py](file:///c:/Users/HP/OneDrive/Desktop/wetLand_Depreciation/src/gis_engine/output/geotiff.py)
- Implement `write_confidence_map(output_path: str, confidence_map: np.ndarray, metadata: RasterMetadata)`: Saves the float32 confidence map to a georeferenced GeoTIFF.
- Implement `write_mask_map(output_path: str, mask: np.ndarray, metadata: RasterMetadata)`: Saves a boolean mask as a uint8 GeoTIFF (1 for valid, 0 for invalid).

#### [MODIFY] [json.py](file:///c:/Users/HP/OneDrive/Desktop/wetLand_Depreciation/src/gis_engine/output/json.py)
- Implement `write_pair_metadata_json(...)`: Serializes the shared configuration, performance stats, grid geometry, model information (including model SHA-256 hash), overlap percentage, and class statistics into `pair_metadata.json`.

---

### Component: CLI Integration
#### [NEW] [analyze_pair.py](file:///c:/Users/HP/OneDrive/Desktop/wetLand_Depreciation/scripts/analyze_pair.py)
- Create a CLI script supporting arguments:
  - `--t1`: Aligned T1 directory.
  - `--t2`: Aligned T2 directory.
  - `--model`: ONNX model path.
  - `--output`: Output directory.
  - `--config`: Config file path.
- Parse inputs, execute `analyze_pair`, print status, and display the formatted console summary requested in the specification.

---

### Component: Verification & Testing
#### [NEW] [test_two_date_inference.py](file:///c:/Users/HP/OneDrive/Desktop/wetLand_Depreciation/tests/test_two_date_inference.py)
- Implement tests verifying:
  1. Identical grid acceptance.
  2. Mismatched grid rejection (CRS, transform, resolution, shape, bounds).
  3. Correct model input tensor shape `(1, 5, H, W)`.
  4. Correct channel ordering `[NDVI, GNDVI, NDRE, NIR, RedEdge]`.
  5. Correct model output shape `(1, 4, H, W)`.
  6. Correct class argmax.
  7. Correct softmax calculation.
  8. Correct confidence calculation.
  9. Correct class statistics over date-valid and common overlap.
  10. NoData is not counted as Background.
  11. Output geometry equality between T1 and T2 outputs.
  12. Logits are preserved.
  13. Same model configuration is used.
  14. Overlap mask is propagated correctly.
  15. Metadata JSON structure validation.

---

## Verification Plan

### Automated Tests
- Run `python -m pytest tests/test_two_date_inference.py` to verify the new units.
- Run `python -m pytest tests/` to verify all tests in the repository pass.

### Manual Verification
- Run the CLI script on the real aligned dataset in `outputs/alignment/2023-08-14_2023-08-28`:
  ```bash
  python scripts/analyze_pair.py --t1 outputs/alignment/2023-08-14_2023-08-28/t1_aligned --t2 outputs/alignment/2023-08-14_2023-08-28/t2_aligned --model models/best_finetuned_model.onnx --output outputs/inference/2023-08-14_2023-08-28
  ```
- Compare the new T1/T2 class maps and statistics with historical single-date runs if applicable.
- Confirm outputs load correctly and align in QGIS.
