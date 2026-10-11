import os
import numpy as np
import rasterio
from pathlib import Path
from scipy.ndimage import label
from typing import Dict, Any, List, Optional, Tuple
import onnxruntime as ort

class QualityAssuranceEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.flags = []
        self.categories_status = {
            "input": "PASS",
            "model": "PASS",
            "spatial": "PASS",
            "temporal": "PASS",
            "change": "PASS",
            "wetland": "PASS",
            "metrics": "PASS",
            "uncertainty": "PASS"
        }
        
    def add_flag(self, code: str, severity: str, message: str, phase: str, status: str, recommended_action: str):
        """Adds a QA flag and updates category status if needed."""
        flag = {
            "code": code,
            "severity": severity,
            "message": message,
            "phase": phase,
            "status": status,
            "recommended_action": recommended_action
        }
        self.flags.append(flag)
        
        # Determine category from code prefix
        category_map = {
            "QA-001": "input", # Input missing
            "QA-002": "input", # Input geometry mismatch
            "QA-003": "spatial", # NoData inconsistency
            "QA-004": "spatial", # CRS mismatch
            "QA-005": "spatial", # Transform mismatch
            "QA-006": "temporal", # Temporal order invalid
            "QA-007": "temporal", # Phenology warning
            "QA-008": "model", # Unexpected class distribution
            "QA-009": "change", # Unexpected transition dominance
            "QA-010": "change", # CVA outlier
            "QA-011": "wetland", # Wetland boundary unavailable
            "QA-012": "uncertainty", # Independent reference samples unavailable
            "QA-013": "model", # Probability calibration unavailable
            "QA-014": "uncertainty", # Olofsson unavailable
            "QA-015": "metrics", # Area reconciliation failure
            "QA-016": "metrics", # NDR invalid
            "QA-017": "reproducibility", # Output reproducibility failure
            "QA-018": "spatial", # Spatial shift detected
            "QA-019": "spatial", # NoData/Background confusion
            "QA-020": "metrics" # Invalid metric
        }
        
        cat = category_map.get(code, "change")
        
        # Elevate status of categories based on severity
        if severity == "CRITICAL":
            self.categories_status[cat] = "FAIL"
        elif severity == "HIGH" and self.categories_status[cat] != "FAIL":
            self.categories_status[cat] = "FAIL"
        elif severity in ("MEDIUM", "LOW") and self.categories_status[cat] == "PASS":
            self.categories_status[cat] = "WARN"
        elif status == "NOT_AVAILABLE" and self.categories_status[cat] == "PASS":
            # Deferrals are allowed and status remains PASS or WARN, not FAIL
            pass

    # 1. INPUT QA
    def run_input_qa(self, t1_bands: Dict[str, Any], t2_bands: Dict[str, Any], meta_t1: Any, meta_t2: Any) -> Dict[str, Any]:
        """Validates input datasets, bands, geometries, ranges, and VIs."""
        results = {"status": "PASS", "bands": {}}
        required_bands = ["green", "red", "rededge", "nir"]
        
        # Check T1 & T2 bands exist
        for date_name, bands, meta in [("T1", t1_bands, meta_t1), ("T2", t2_bands, meta_t2)]:
            for b in required_bands:
                key = f"{date_name}_{b}"
                if b not in bands or bands[b] is None:
                    self.add_flag(
                        "QA-001", "CRITICAL", f"Required band '{b}' missing for date {date_name}",
                        "Input Ingestion", "FAIL", f"Provide the missing {b} band for {date_name}."
                    )
                    results["status"] = "FAIL"
                    continue
                
                arr = bands[b]
                results["bands"][key] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "mean": float(np.nanmean(arr)),
                    "median": float(np.nanmedian(arr)),
                    "p1": float(np.percentile(arr, 1)),
                    "p99": float(np.percentile(arr, 99))
                }
                
                # Check for extreme/negative values
                if np.any(arr < 0.0):
                    self.add_flag(
                        "QA-010", "MEDIUM", f"Negative values detected in band '{b}' for {date_name}",
                        "Input Ingestion", "WARN", f"Check reflectance values in {b} band."
                    )
                # Check for NaN / Inf
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    self.add_flag(
                        "QA-003", "HIGH", f"NaN or Inf values found in raw input band '{b}' for {date_name}",
                        "Input Ingestion", "WARN", "Verify preprocessing handles NaNs/Infs."
                    )
                # Check constant band
                if np.nanmax(arr) - np.nanmin(arr) < 1e-6:
                    self.add_flag(
                        "QA-010", "HIGH", f"Constant values detected in band '{b}' for {date_name}",
                        "Input Ingestion", "WARN", "Confirm sensor data is valid."
                    )

        # Check geometry comparability within dates
        for date_name, bands, meta in [("T1", t1_bands, meta_t1), ("T2", t2_bands, meta_t2)]:
            if meta is None:
                continue
            ref_shape = bands["green"].shape if "green" in bands else None
            for b in required_bands:
                if b in bands and bands[b] is not None:
                    if bands[b].shape != ref_shape:
                        self.add_flag(
                            "QA-002", "CRITICAL", f"Shape mismatch in {date_name} band {b} vs green",
                            "Input Ingestion", "FAIL", "Align raw bands using Phase 2 aligner."
                        )
                        results["status"] = "FAIL"
                        
        return results

    def run_vegetation_index_qa(self, t1_indices: Dict[str, Any], t2_indices: Dict[str, Any], valid_mask: np.ndarray) -> Dict[str, Any]:
        """Verifies mathematical sanity of indices."""
        res = {}
        for name, indices in [("T1", t1_indices), ("T2", t2_indices)]:
            res[name] = {}
            for idx in ["ndvi", "gndvi", "ndre"]:
                if idx not in indices or indices[idx] is None:
                    continue
                arr = indices[idx][valid_mask]
                if len(arr) == 0:
                    continue
                
                obs_min = float(np.min(arr))
                obs_max = float(np.max(arr))
                obs_mean = float(np.mean(arr))
                obs_median = float(np.median(arr))
                p5 = float(np.percentile(arr, 5))
                p95 = float(np.percentile(arr, 95))
                
                res[name][idx] = {
                    "min": obs_min, "max": obs_max, "mean": obs_mean,
                    "median": obs_median, "p5": p5, "p95": p95
                }
                
                # Check theoretical range
                if obs_min < -1.05 or obs_max > 1.05:
                    self.add_flag(
                        "QA-020", "MEDIUM", f"{name} {idx.upper()} range [{obs_min:.2f}, {obs_max:.2f}] violates theoretical range [-1, 1]",
                        "Preprocessing", "WARN", "Verify index computation formula."
                    )
        return res

    # 2. MODEL QA
    def run_model_qa(self, model_path: str, logits: Optional[np.ndarray], probabilities: Optional[np.ndarray]) -> Dict[str, Any]:
        """Validates model properties, logits, softmax summation, and confidence levels."""
        res = {"status": "PASS"}
        
        # Model file check
        if not os.path.isfile(model_path):
            self.add_flag(
                "QA-001", "CRITICAL", f"ONNX model file not found at {model_path}",
                "Model QA", "FAIL", "Deploy the correct ONNX model."
            )
            res["status"] = "FAIL"
            return res
            
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_details = session.get_inputs()[0]
            output_details = session.get_outputs()[0]
            res["onnx"] = {
                "input_name": input_details.name,
                "input_shape": input_details.shape,
                "output_name": output_details.name,
                "output_shape": output_details.shape,
                "providers": session.get_providers()
            }
            
            # Check model channel contract (must be 5)
            if input_details.shape[1] != 5:
                self.add_flag(
                    "QA-002", "CRITICAL", f"ONNX model expects {input_details.shape[1]} input channels. Expected 5.",
                    "Model QA", "FAIL", "Rebuild input tensor to match channel order NDVI, GNDVI, NDRE, NIR, RedEdge."
                )
                res["status"] = "FAIL"
        except Exception as e:
            self.add_flag(
                "QA-001", "CRITICAL", f"Failed to load ONNX model: {str(e)}",
                "Model QA", "FAIL", "Check model format and integrity."
            )
            res["status"] = "FAIL; " + str(e)
            return res

        # Probabilities contract check
        if probabilities is not None:
            sums = np.sum(probabilities, axis=0)
            sums_valid = np.allclose(sums, 1.0, atol=1e-4)
            prob_range = np.all((probabilities >= 0.0) & (probabilities <= 1.0))
            if not (sums_valid and prob_range):
                self.add_flag(
                    "QA-020", "HIGH", "Model output probabilities violate sum ≈ 1 or [0, 1] range.",
                    "Model Inference", "FAIL", "Check softmax application code."
                )
                res["status"] = "FAIL"
                
            # Confidence statistics
            max_probs = np.max(probabilities, axis=0)
            res["confidence"] = {
                "mean_max_prob": float(np.mean(max_probs)),
                "median_max_prob": float(np.median(max_probs)),
                "p5_max_prob": float(np.percentile(max_probs, 5)),
                "p95_max_prob": float(np.percentile(max_probs, 95))
            }
        else:
            self.add_flag(
                "QA-013", "INFO", "Probability outputs are unavailable. Reporting calibration as NOT_AVAILABLE.",
                "Model Inference", "NOT_AVAILABLE", "Calibration deferred to V2."
            )
            res["calibration_status"] = "NOT_AVAILABLE"
            res["confidence"] = "NOT_AVAILABLE"

        return res

    def run_class_distribution_qa(self, t1_health: np.ndarray, t2_health: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
        """Checks for extreme class distributions."""
        res = {}
        for name, arr in [("T1", t1_health), ("T2", t2_health)]:
            valid_arr = arr[valid_mask]
            total_pixels = len(valid_arr)
            if total_pixels == 0:
                continue
            
            classes, counts = np.unique(valid_arr, return_counts=True)
            dist = {int(c): float(cnt / total_pixels * 100.0) for c, cnt in zip(classes, counts)}
            res[name] = dist
            
            # Warn if single class dominates > 99%
            for c, pct in dist.items():
                if pct > 99.0:
                    self.add_flag(
                        "QA-008", "MEDIUM", f"Class {c} dominates {name} predictions with {pct:.2f}% of pixels",
                        "Model Inference", "WARN", "Confirm this dominance reflects true scene conditions."
                    )
        return res

    # 3. SPATIAL QA
    def run_spatial_qa(self, raster_metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Checks spatial alignment, CRS, transform, resolution, and shift consistency."""
        res = {"status": "PASS"}
        if not raster_metadata_dict:
            return res
            
        ref_key = list(raster_metadata_dict.keys())[0]
        ref = raster_metadata_dict[ref_key]
        
        for k, meta in raster_metadata_dict.items():
            if meta is None:
                continue
            crs_ok = (meta.crs == ref.crs)
            trans_ok = (meta.transform == ref.transform)
            shape_ok = (meta.width == ref.width and meta.height == ref.height)
            res_ok = (meta.resolution == ref.resolution)
            
            meta_bounds = getattr(meta, "bounds", getattr(meta, "extent", None))
            ref_bounds = getattr(ref, "bounds", getattr(ref, "extent", None))
            bounds_ok = (meta_bounds == ref_bounds)
            
            if not crs_ok:
                self.add_flag(
                    "QA-004", "CRITICAL", f"CRS mismatch on raster '{k}' vs reference '{ref_key}'",
                    "Spatial Alignment", "FAIL", "Reproject outputs to the common reference grid."
                )
                res["status"] = "FAIL"
            if not trans_ok or not res_ok:
                self.add_flag(
                    "QA-005", "CRITICAL", f"Transform/Resolution mismatch on raster '{k}'",
                    "Spatial Alignment", "FAIL", "Re-align using Phase 2 aligner."
                )
                res["status"] = "FAIL"
            if not shape_ok:
                self.add_flag(
                    "QA-002", "CRITICAL", f"Dimensions mismatch on raster '{k}' ({meta.height}x{meta.width} vs {ref.height}x{ref.width})",
                    "Spatial Alignment", "FAIL", "Resample raster dimensions."
                )
                res["status"] = "FAIL"
                
        # Spatial shift test (Marker check / verification)
        # Note: Simulated shift test returns PASS if shapes and transforms are identical
        shift_detected = False
        for k, meta in raster_metadata_dict.items():
            if meta is None:
                continue
            # Check if offsets are not aligned to pixel increments
            dx = meta.transform[2] - ref.transform[2]
            dy = meta.transform[5] - ref.transform[5]
            if abs(dx) > 1e-7 or abs(dy) > 1e-7:
                shift_detected = True
                
        if shift_detected:
            self.add_flag(
                "QA-018", "CRITICAL", "Spatial shift/pixel offset detected across outputs",
                "Spatial Alignment", "FAIL", "Recalibrate the affine transformation alignment."
            )
            res["status"] = "FAIL"
            
        return res

    def run_nodata_propagation_qa(self, t1_arr: np.ndarray, t2_arr: np.ndarray, outputs: Dict[str, np.ndarray], valid_overlap: np.ndarray) -> Dict[str, Any]:
        """Verifies that NoData values remain NoData and do not contaminate other values."""
        res = {"status": "PASS"}
        
        # Pixels that are NoData in either T1 or T2
        nodata_mask = (~valid_overlap)
        if not np.any(nodata_mask):
            return res
            
        # Verify all outputs are NoData inside the nodata_mask
        for out_name, arr in outputs.items():
            if arr is None:
                continue
            
            # For categorical rasters, NoData is 255
            # For CVA/float, NoData is NaN
            nodata_pixels = arr[nodata_mask]
            
            if np.issubdtype(arr.dtype, np.integer):
                # check that none are valid classes (0, 1, 2, 3)
                corrupted = np.any(np.isin(nodata_pixels, [0, 1, 2, 3]))
                if corrupted:
                    self.add_flag(
                        "QA-003", "CRITICAL", f"NoData pixels converted to valid class in categorical raster '{out_name}'",
                        "NoData QA", "FAIL", "Enforce NoData mask propagation at output write."
                    )
                    res["status"] = "FAIL"
            else:
                # float rasters must be NaN
                corrupted = np.any(np.isfinite(nodata_pixels) & (nodata_pixels == 0.0))
                if corrupted:
                    self.add_flag(
                        "QA-003", "CRITICAL", f"NoData pixels converted to zero value in float raster '{out_name}'",
                        "NoData QA", "FAIL", "Ensure NoData values propagate as NaNs."
                    )
                    res["status"] = "FAIL"
                    
        # Background (0) vs NoData (255) Confusion Test
        # Ensure Background does not get replaced by NoData or vice versa
        for date_name, arr in [("T1", t1_arr), ("T2", t2_arr)]:
            bg_pixels = (arr == 0) & valid_overlap
            if np.any(bg_pixels) and np.any(arr[bg_pixels] == 255):
                self.add_flag(
                    "QA-019", "CRITICAL", f"Background vs NoData confusion detected in {date_name} health map",
                    "NoData QA", "FAIL", "Ensure Background class 0 is distinct from NoData 255."
                )
                res["status"] = "FAIL"
                
        return res

    # 4. TEMPORAL QA
    def run_temporal_qa(self, t1_date: str, t2_date: str, meta_t1: Any, meta_t2: Any, valid_overlap_mask: np.ndarray) -> Dict[str, Any]:
        """Performs acquisition chronology, seasonality, and overlap size validation."""
        res = {}
        try:
            from datetime import datetime
            d1 = datetime.strptime(t1_date, "%Y-%m-%d")
            d2 = datetime.strptime(t2_date, "%Y-%m-%d")
            
            diff_days = (d2 - d1).days
            res["date_difference_days"] = diff_days
            
            if diff_days <= 0:
                self.add_flag(
                    "QA-006", "CRITICAL", f"Invalid chronology: T2 ({t2_date}) is not after T1 ({t1_date})",
                    "Temporal Alignment", "FAIL", "Verify temporal directory inputs."
                )
                
            # Phenology Warning
            # Check seasonal/month gap (e.g. comparing spring vs autumn)
            month_diff = abs(d1.month - d2.month)
            # If date difference is not close to multi-year or is a seasonal transition
            is_seasonal_mismatch = (3 <= month_diff <= 9)
            if is_seasonal_mismatch:
                self.add_flag(
                    "QA-007", "MEDIUM", f"Acquisitions are in different seasons (month diff: {month_diff} months)",
                    "Temporal Alignment", "WARN", "Review vegetation index values for seasonal/phenological bias."
                )
                res["phenology_warning"] = "YES"
            else:
                res["phenology_warning"] = "NO"
        except Exception as e:
            res["error"] = str(e)
            
        # Overlap QA
        total_t1 = float(np.sum(t1_bands_exist_mask := np.ones_like(valid_overlap_mask))) # placeholder
        overlap_pixels = float(np.sum(valid_overlap_mask))
        res["valid_overlap_pixels"] = overlap_pixels
        
        if overlap_pixels == 0:
            self.add_flag(
                "QA-002", "CRITICAL", "Valid overlap area is zero pixels",
                "Spatial Alignment", "FAIL", "Check spatial bounds overlap between dates."
            )
            
        return res

    # 5. TRANSITION QA
    def run_transition_qa(self, transition_map: np.ndarray, t1_health: np.ndarray, t2_health: np.ndarray, valid_overlap: np.ndarray) -> Dict[str, Any]:
        """Validates transitions mapping IDs, totals, and reconciliation."""
        res = {"status": "PASS"}
        mask = valid_overlap & (t1_health != 255) & (t2_health != 255)
        
        # Verify formula: trans = T1 * 4 + T2
        expected_trans = t1_health[mask].astype(np.int32) * 4 + t2_health[mask].astype(np.int32)
        actual_trans = transition_map[mask].astype(np.int32)
        
        formula_correct = np.array_equal(expected_trans, actual_trans)
        if not formula_correct:
            self.add_flag(
                "QA-020", "CRITICAL", "Transition ID calculation formula mismatch.",
                "Transition Mapping", "FAIL", "Verify formula: transition_id = T1 * 4 + T2."
            )
            res["status"] = "FAIL"
            
        # Reconciliation check
        total_trans_pixels = float(np.sum(mask))
        reported_pixels = float(np.sum(np.isin(transition_map, list(range(16))) & valid_overlap))
        if abs(total_trans_pixels - reported_pixels) > 0:
            self.add_flag(
                "QA-015", "CRITICAL", "Transition sum reconciliation failure",
                "Transition Mapping", "FAIL", "Confirm transition raster counts sum to total overlap area."
            )
            res["status"] = "FAIL"
            
        # Dominance check (e.g. > 99% stable background or stable healthy)
        classes, counts = np.unique(transition_map[mask], return_counts=True)
        total_p = len(transition_map[mask])
        for c, cnt in zip(classes, counts):
            pct = cnt / total_p * 100.0
            if pct > 99.0:
                self.add_flag(
                    "QA-009", "WARN", f"Transition ID {c} dominates transition map with {pct:.2f}% of pixels",
                    "Transition Mapping", "WARN", "Verify change detection sensitivity settings."
                )
                
        return res

    # 6. CHANGE PATCH QA
    def run_change_patch_qa(self, degradation: np.ndarray, recovery: np.ndarray, canopy_loss: np.ndarray, residual_stress: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
        """Calculates connected components statistics for the maps."""
        res = {}
        for name, arr in [("degradation", degradation), ("recovery", recovery), ("canopy_loss", canopy_loss), ("residual_stress", residual_stress)]:
            if arr is None:
                continue
            # Mask out background/nodata
            change_pixels = (arr == 1) & valid_mask
            
            # Connected components
            labeled_arr, num_features = label(change_pixels)
            if num_features > 0:
                unique_lbl, sizes = np.unique(labeled_arr, return_counts=True)
                # Exclude label 0 (background)
                sizes = sizes[1:]
                res[name] = {
                    "patch_count": int(num_features),
                    "min_size_px": int(np.min(sizes)),
                    "median_size_px": int(np.median(sizes)),
                    "max_size_px": int(np.max(sizes))
                }
            else:
                res[name] = {
                    "patch_count": 0, "min_size_px": 0, "median_size_px": 0, "max_size_px": 0
                }
        return res

    # 7. CVA QA
    def run_cva_qa(self, cva_mag: np.ndarray, d_ndvi: np.ndarray, d_gndvi: np.ndarray, d_ndre: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
        """Checks math identity and outliers of CVA."""
        res = {"status": "PASS"}
        mask = valid_mask & np.isfinite(cva_mag)
        if not np.any(mask):
            return res
            
        cva_mag_sq = cva_mag[mask] ** 2
        sum_delta_sq = d_ndvi[mask]**2 + d_gndvi[mask]**2 + d_ndre[mask]**2
        
        diff = np.abs(cva_mag_sq - sum_delta_sq)
        max_err = float(np.max(diff))
        
        res["max_reconciliation_error"] = max_err
        if max_err > 1e-4:
            self.add_flag(
                "QA-020", "CRITICAL", f"CVA quadratic math identity check failed (Max err: {max_err})",
                "CVA Math Invariant", "FAIL", "Check CVA magnitude computation formula."
            )
            res["status"] = "FAIL"
            
        # Outlier QA
        m_mag = cva_mag[mask]
        res["outliers"] = {
            "min": float(np.min(m_mag)),
            "max": float(np.max(m_mag)),
            "mean": float(np.mean(m_mag)),
            "std": float(np.std(m_mag)),
            "p1": float(np.percentile(m_mag, 1)),
            "p99": float(np.percentile(m_mag, 99))
        }
        
        if res["outliers"]["max"] > 10.0:
            self.add_flag(
                "QA-010", "MEDIUM", f"Extreme CVA magnitude outlier detected: max CVA = {res['outliers']['max']:.2f}",
                "CVA Engine", "WARN", "Verify raw band scaling and index computation inputs."
            )
            
        return res

    # 8. WETLAND MASK QA
    def run_wetland_qa(self, wetland_mask: Optional[np.ndarray], valid_analysis_mask: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validates wetland source, membership values, and claims bounds."""
        res = {}
        
        wetland_cfg = config.get("wetland", {})
        source_type = wetland_cfg.get("source", "auto")
        boundary_path = wetland_cfg.get("boundary", {}).get("path")
        
        if boundary_path is not None:
            source_status = "WETLAND_BOUNDARY_AVAILABLE"
        elif source_type == "inferred":
            source_status = "WETLAND_EXTENT_INFERRED"
        else:
            source_status = "WETLAND_BOUNDARY_UNAVAILABLE"
            self.add_flag(
                "QA-011", "WARN", "Authoritative wetland boundary unavailable. Physical wetland depletion calculations disabled.",
                "Wetland Analysis", "WARN", "Configure official wetland boundary dataset path in config.json."
            )
            
        res["source_status"] = source_status
        
        # Verify membership values
        if wetland_mask is not None:
            unique_vals = np.unique(wetland_mask[valid_analysis_mask == 1])
            invalid_vals = [int(v) for v in unique_vals if v not in (0, 1, 255)]
            if len(invalid_vals) > 0:
                self.add_flag(
                    "QA-003", "CRITICAL", f"Invalid values {invalid_vals} found in wetland mask.",
                    "Wetland Analysis", "FAIL", "Ensure wetland mask values are strictly 0 (outside), 1 (inside), or 255 (NoData)."
                )
                
        # Safeguard physical depletion claim bounds
        depletion_claims_cfg = wetland_cfg.get("depletion_claims", {}).get("enabled", False)
        if depletion_claims_cfg and source_status == "WETLAND_BOUNDARY_UNAVAILABLE":
            self.add_flag(
                "QA-011", "CRITICAL", "Wetland depletion claims enabled but no official boundary available.",
                "Wetland Analysis", "FAIL", "Disable physical depletion claims or provide official boundary."
            )
            
        return res

    # 9. METRICS QA
    def run_metrics_qa(self, metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Validates baseline bounds, weights status, and NDR ranges."""
        res = {"status": "PASS"}
        
        baseline_area = metrics.get("baseline", {}).get("area_ha", 0.0)
        impact_area = metrics.get("degradation", {}).get("impact_area_ha", 0.0)
        recovery_area = metrics.get("recovery", {}).get("area_ha", 0.0)
        canopy_loss_area = metrics.get("canopy_loss", {}).get("area_ha", 0.0)
        residual_stress_area = metrics.get("residual_stress", {}).get("area_ha", 0.0)
        
        # Areas >= 0
        if baseline_area < 0 or impact_area < 0 or recovery_area < 0 or canopy_loss_area < 0 or residual_stress_area < 0:
            self.add_flag(
                "QA-020", "CRITICAL", "Negative area detected in output metrics",
                "Metrics QA", "FAIL", "Check spatial pixel area calculation routines."
            )
            res["status"] = "FAIL"
            
        # Impact <= Baseline, etc.
        if impact_area > baseline_area:
            self.add_flag(
                "QA-020", "CRITICAL", f"Impact area ({impact_area:.2f} ha) exceeds baseline area ({baseline_area:.2f} ha)",
                "Metrics QA", "FAIL", "Check baseline mask and class subset bounds."
            )
            res["status"] = "FAIL"
            
        # Normalized Degradation Ratio (NDR) range bounds [0, 1]
        ndr = metrics.get("degradation", {}).get("normalized_degradation_ratio")
        if ndr is not None:
            if ndr < 0.0 or ndr > 1.0:
                self.add_flag(
                    "QA-016", "CRITICAL", f"NDR value {ndr:.4f} is outside valid bounds [0, 1]",
                    "Metrics QA", "FAIL", "Confirm severity-weighted area does not exceed baseline."
                )
                res["status"] = "FAIL"
        elif baseline_area == 0.0:
            self.add_flag(
                "QA-016", "WARN", "Baseline area is zero. NDR is null.",
                "Metrics QA", "WARN", "Provide a non-empty analysis overlap area."
            )

        # Severity weights check
        weights = config.get("metrics", {}).get("severity_weights", {})
        standard_weights = {
            "healthy": 0.0,
            "moderate": 0.35,
            "severe": 0.80,
            "background": 1.0
        }
        is_provisional = all(weights.get(k) == standard_weights[k] for k in standard_weights)
        if is_provisional:
            res["severity_weights_disclaimer"] = "PROVISIONAL_ENGINEERING_PARAMETERS"
            
        return res

    # 10. UNCERTAINTY QA
    def run_uncertainty_qa(self, config: Dict[str, Any], reference_samples_exist: bool) -> Dict[str, Any]:
        """Separates and rates uncertainty categories."""
        res = {}
        
        # 1. Model Uncertainty
        res["model"] = "NOT_AVAILABLE" # logits only by default in V1
        
        # 2. Classification Uncertainty
        res["classification"] = "PARTIAL" # based on class distributions and thresholds
        
        # 3. Spatial Uncertainty
        res["spatial"] = "AVAILABLE" # pixel dimensions and grid resolution are georeferenced
        
        # 4. Temporal Uncertainty
        res["temporal"] = "AVAILABLE" # date gap is known and reported
        
        # 5. Wetland Boundary Uncertainty
        wetland_cfg = config.get("wetland", {})
        boundary_path = wetland_cfg.get("boundary", {}).get("path")
        if boundary_path is not None:
            res["wetland_boundary"] = "LOWER_BOUNDARY_UNCERTAINTY"
        elif wetland_cfg.get("source") == "inferred":
            res["wetland_boundary"] = "INFERRED_BOUNDARY"
        else:
            res["wetland_boundary"] = "NOT_AVAILABLE"
            self.add_flag(
                "QA-011", "LOW", "Wetland boundary is inferred or unavailable",
                "Uncertainty QA", "WARN", "Use authoritative datasets to reduce wetland boundary uncertainty."
            )
            
        # 6. Area Estimation Uncertainty
        if reference_samples_exist:
            res["area_estimation"] = "AVAILABLE"
        else:
            res["area_estimation"] = "NOT_AVAILABLE"
            self.add_flag(
                "QA-012", "INFO", "Independent reference samples unavailable. Area-estimation uncertainty unavailable.",
                "Uncertainty QA", "NOT_AVAILABLE", "Collect reference validation samples for V2."
            )
            self.add_flag(
                "QA-014", "INFO", "Olofsson area-error correction unavailable.",
                "Uncertainty QA", "NOT_AVAILABLE", "Olofsson validation requires independent reference data."
            )

        return res
        
    def generate_qa_report(self) -> Dict[str, Any]:
        """Aggregates statuses and flags into a final QA dictionary."""
        # Find overall status (FAIL if any critical failures exist)
        overall = "PASS"
        if any(f["severity"] == "CRITICAL" for f in self.flags if f["status"] == "FAIL"):
            overall = "FAIL"
        elif any(f["severity"] == "HIGH" for f in self.flags if f["status"] == "FAIL"):
            overall = "FAIL"
        elif any(f["status"] == "WARN" for f in self.flags):
            overall = "WARN"
            
        return {
            "overall_status": overall,
            "categories": self.categories_status,
            "flags": self.flags
        }
