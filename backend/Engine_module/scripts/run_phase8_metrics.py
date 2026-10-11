import os
import sys
import json
import csv
import numpy as np
from pathlib import Path

# Ensure src/ is on the path so we can import gis_engine
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gis_engine.metrics.degradation import calculate_metrics

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    print("=== GIS Wetland Degradation Engine - Phase 8 Metrics ===")

    # 1. Load config
    config_path = Path("configs/config.json")
    if not config_path.is_file():
        print(f"Error: Configuration file not found at: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Setup paths
    pair_name = "2023-08-14_2023-08-28"
    
    transition_map = Path("outputs/transitions") / pair_name / "transition_map.tif"
    cva_magnitude = Path("outputs/cva") / pair_name / "cva_magnitude.tif"
    delta_ndvi = Path("outputs/cva") / pair_name / "delta_ndvi.tif"
    delta_gndvi = Path("outputs/cva") / pair_name / "delta_gndvi.tif"
    delta_ndre = Path("outputs/cva") / pair_name / "delta_ndre.tif"
    
    degradation_map = Path("outputs/degradation") / pair_name / "degradation_map.tif"
    recovery_map = Path("outputs/degradation") / pair_name / "recovery_map.tif"
    residual_stress = Path("outputs/degradation") / pair_name / "residual_stress.tif"
    canopy_loss = Path("outputs/degradation") / pair_name / "canopy_loss_map.tif"
    
    wetland_mask = Path("outputs/wetland") / "wetland_mask.tif"
    valid_analysis_mask = Path("outputs/wetland") / "valid_wetland_analysis_mask.tif"

    # Verify input files
    input_paths = [
        transition_map, cva_magnitude, delta_ndvi, delta_gndvi, delta_ndre,
        degradation_map, recovery_map, residual_stress, canopy_loss,
        wetland_mask, valid_analysis_mask
    ]

    for p in input_paths:
        if not p.is_file():
            print(f"Error: Required input file does not exist: {p}")
            sys.exit(1)

    print("All input rasters verified. Running Phase 8 Metrics calculations...")

    # 3. Calculate metrics
    res = calculate_metrics(
        transition_map=transition_map,
        cva_magnitude=cva_magnitude,
        delta_ndvi=delta_ndvi,
        delta_gndvi=delta_gndvi,
        delta_ndre=delta_ndre,
        degradation_map=degradation_map,
        recovery_map=recovery_map,
        residual_stress_map=residual_stress,
        canopy_loss_map=canopy_loss,
        wetland_mask=wetland_mask,
        valid_analysis_mask=valid_analysis_mask,
        config=config
    )

    print("Metrics calculation completed successfully. Saving outputs...")

    # 4. Save outputs to outputs/metrics/
    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json structure
    metrics_json_content = {
        "baseline": res.baseline_area,
        "degradation": res.impact_area,
        "recovery": res.recovery_area,
        "canopy_loss": res.canopy_loss_area,
        "residual_stress": res.residual_stress_area,
        "class_areas": res.class_area_metrics,
        "transition_areas": res.transition_metrics,
        "cva": res.cva_statistics,
        "wetland_extent": res.wetland_extent_metrics,
        "qa": res.qa_flags,
        "metadata": res.metadata
    }

    # statistics.json structure
    statistics_json_content = {
        "class_distributions": res.class_area_metrics,
        "transition_distributions": res.transition_metrics,
        "cva_distributions": res.cva_statistics,
        "delta_index_distributions": res.delta_statistics,
        "area_metrics": {
            "baseline": res.baseline_area,
            "degradation": res.impact_area,
            "recovery": res.recovery_area,
            "canopy_loss": res.canopy_loss_area,
            "residual_stress": res.residual_stress_area
        },
        "qa_flags": res.qa_flags
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json_content, f, indent=4)
    print(f"Saved: {output_dir / 'metrics.json'}")

    with open(output_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics_json_content, f, indent=4)
    print(f"Saved: {output_dir / 'statistics.json'}")

    # Generate metrics.csv
    csv_rows = []
    # Flat summary metrics
    csv_rows.append(["metric_name", "value", "units"])
    csv_rows.append(["baseline_area", res.baseline_area["area_ha"], "ha"])
    csv_rows.append(["baseline_area_m2", res.baseline_area["area_m2"], "m2"])
    csv_rows.append(["affected_area", res.impact_area["impact_area_ha"], "ha"])
    csv_rows.append(["severity_weighted_area", res.impact_area["severity_weighted_area_ha_deg"], "ha_deg"])
    csv_rows.append(["normalized_degradation_ratio", res.normalized_degradation_ratio if res.normalized_degradation_ratio is not None else "NoData", ""])
    csv_rows.append(["recovery_area", res.recovery_area["area_ha"], "ha"])
    csv_rows.append(["residual_stress_area", res.residual_stress_area["area_ha"], "ha"])
    csv_rows.append(["canopy_loss_area", res.canopy_loss_area["area_ha"], "ha"])
    csv_rows.append(["mean_cva", res.cva_statistics["all_valid_wetland"]["mean"], ""])
    csv_rows.append(["median_cva", res.cva_statistics["all_valid_wetland"]["median"], ""])
    csv_rows.append(["p95_cva", res.cva_statistics["all_valid_wetland"]["p95"], ""])
    csv_rows.append(["wetland_extent_status", res.wetland_extent_metrics["status"], ""])

    with open(output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Saved: {output_dir / 'metrics.csv'}")

    # Print summary of the results to stdout
    print("\n================== SUMMARY REPORT ==================")
    print(f"Baseline Wetland Area:          {res.baseline_area['area_ha']:.4f} ha")
    print(f"Affected / Degraded Area:       {res.impact_area['impact_area_ha']:.4f} ha")
    print(f"Severity-Weighted Degraded Area:{res.impact_area['severity_weighted_area_ha_deg']:.4f} ha_deg")
    print(f"Normalized Degradation Ratio:   {res.normalized_degradation_ratio if res.normalized_degradation_ratio is not None else 'NoData'}")
    print(f"Recovery Area:                  {res.recovery_area['area_ha']:.4f} ha")
    print(f"Residual Stress Area:           {res.residual_stress_area['area_ha']:.4f} ha")
    print(f"Canopy-Loss Area:               {res.canopy_loss_area['area_ha']:.4f} ha")
    print(f"Wetland Extent Status:          {res.wetland_extent_metrics['status']}")
    print(f"Reconciliation Transition:      {'PASS' if res.qa_flags['flags']['reconciliation_transition_passes'] else 'FAIL'}")
    print(f"Reconciliation Class:           {'PASS' if res.qa_flags['flags']['reconciliation_class_passes'] else 'FAIL'}")
    print("====================================================")


if __name__ == "__main__":
    main()
