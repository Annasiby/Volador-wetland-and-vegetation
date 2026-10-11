import numpy as np
from typing import Dict, Any, Optional, List

def estimate_area_with_reference_samples(
    classification_map: Any,
    reference_samples: Optional[List[Dict[str, Any]]] = None,
    strata_definition: Optional[Dict[int, Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Interface and mathematical implementation of Olofsson area estimation
    using independent reference samples (Olofsson et al., 2014).
    
    If no reference samples exist or are provided, returns status = "NOT_AVAILABLE"
    and does not fabricate confidence intervals or reference accuracies.
    
    Args:
        classification_map: 2D numpy array of class predictions, or None.
        reference_samples: Optional list of sample dicts containing 'mapped_class' and 'reference_class'.
        strata_definition: Optional dictionary defining mapped area per stratum.
        config: Optional configuration dictionary.
        
    Returns:
        Dict containing Olofsson area estimates, proportions, variance, and confidence intervals.
    """
    if reference_samples is None or len(reference_samples) == 0:
        return {
            "status": "NOT_AVAILABLE",
            "model_derived_estimate": True,
            "independent_area_error_correction": False,
            "message": "Model-derived estimate; no independent area-error correction applied."
        }

    try:
        # Extract class IDs from classification map or strata definitions
        if classification_map is not None:
            unique_classes = np.unique(classification_map)
            unique_classes = [int(c) for c in unique_classes if c != 255] # exclude NoData
        elif strata_definition is not None:
            unique_classes = list(strata_definition.keys())
        else:
            # Infer from reference samples
            unique_classes = list(set([s["mapped_class"] for s in reference_samples] + 
                                      [s["reference_class"] for s in reference_samples]))
            
        # Get total area/pixel count per stratum
        strata_areas = {}
        total_area = 0.0
        
        if strata_definition is not None:
            for c in unique_classes:
                strata_areas[c] = strata_definition.get(c, {}).get("area", 0.0)
                total_area += strata_areas[c]
        elif classification_map is not None:
            for c in unique_classes:
                count = float(np.sum(classification_map == c))
                strata_areas[c] = count
                total_area += count
        else:
            # If no areas are available, assume equal weight for simplicity or use counts
            for c in unique_classes:
                strata_areas[c] = 1.0 / len(unique_classes)
            total_area = 1.0

        if total_area <= 0:
            return {
                "status": "NOT_AVAILABLE",
                "message": "Total analysis area is zero. Cannot compute Olofsson area estimation."
            }

        # Calculate area weights Wh
        Wh = {c: strata_areas[c] / total_area for c in unique_classes}
        
        # Build confusion matrix from reference samples
        # rows: mapped classes (strata h), columns: reference classes (j)
        nh_dot = {c: 0 for c in unique_classes}
        nhj = {h: {j: 0 for j in unique_classes} for h in unique_classes}
        
        for sample in reference_samples:
            h = sample.get("mapped_class")
            j = sample.get("reference_class")
            if h in unique_classes and j in unique_classes:
                nh_dot[h] += 1
                nhj[h][j] += 1
                
        # Calculate estimated proportions p_j
        # p_j = sum_h Wh * (nhj / nh_dot)
        p_j = {j: 0.0 for j in unique_classes}
        for j in unique_classes:
            for h in unique_classes:
                if nh_dot[h] > 0:
                    p_j[j] += Wh[h] * (nhj[h][j] / nh_dot[h])
                    
        # Calculate standard error of estimated proportions
        # S(p_j) = sqrt( sum_h (Wh^2 * (nhj/nh_dot) * (1 - nhj/nh_dot)) / (nh_dot - 1) )
        se_p_j = {j: 0.0 for j in unique_classes}
        for j in unique_classes:
            variance_sum = 0.0
            for h in unique_classes:
                n = nh_dot[h]
                if n > 1:
                    prop = nhj[h][j] / n
                    variance_sum += (Wh[h] ** 2 * prop * (1.0 - prop)) / (n - 1.0)
            se_p_j[j] = np.sqrt(variance_sum)

        # Calculate adjusted area estimates and confidence intervals (95% CI -> z = 1.96)
        adjusted_areas = {}
        for c in unique_classes:
            est_area = p_j[c] * total_area
            se_area = se_p_j[c] * total_area
            ci_half = 1.96 * se_area
            adjusted_areas[c] = {
                "stratum": c,
                "mapped_area": strata_areas[c],
                "mapped_proportion": Wh[c],
                "adjusted_area_estimate": est_area,
                "adjusted_proportion_estimate": p_j[c],
                "standard_error_proportion": se_p_j[c],
                "standard_error_area": se_area,
                "confidence_interval_95_lower": max(0.0, est_area - ci_half),
                "confidence_interval_95_upper": est_area + ci_half
            }

        return {
            "status": "PASS",
            "model_derived_estimate": False,
            "independent_area_error_correction": True,
            "total_area": total_area,
            "sample_counts": nh_dot,
            "strata_results": adjusted_areas
        }
    except Exception as e:
        return {
            "status": "NOT_AVAILABLE",
            "model_derived_estimate": True,
            "independent_area_error_correction": False,
            "message": f"Olofsson estimation calculation failed: {str(e)}"
        }
