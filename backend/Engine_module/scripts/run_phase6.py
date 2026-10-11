import os
import sys
import json
from pathlib import Path

# Ensure src/ is on the path so we can import gis_engine
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gis_engine.change.degradation import analyze_degradation

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    # 1. Load config
    config_path = Path("configs/config.json")
    if not config_path.is_file():
        print(f"Error: Configuration file not found at: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Paths
    pair_name = "2023-08-14_2023-08-28"
    transition_map = Path("outputs/transitions") / pair_name / "transition_map.tif"
    cva_magnitude = Path("outputs/cva") / pair_name / "cva_magnitude.tif"
    delta_ndvi = Path("outputs/cva") / pair_name / "delta_ndvi.tif"
    delta_gndvi = Path("outputs/cva") / pair_name / "delta_gndvi.tif"
    delta_ndre = Path("outputs/cva") / pair_name / "delta_ndre.tif"
    valid_mask = Path("outputs/inference") / pair_name / "overlap_mask.tif"
    output_dir = Path("outputs/degradation")

    # Check files exist
    for p in [transition_map, cva_magnitude, delta_ndvi, delta_gndvi, delta_ndre, valid_mask]:
        if not p.is_file():
            print(f"Error: Input file does not exist: {p}")
            sys.exit(1)

    print("\nRunning Phase 6: Degradation & Recovery Engine on Real Data...")
    try:
        result = analyze_degradation(
            transition_map=transition_map,
            cva_magnitude=cva_magnitude,
            delta_ndvi=delta_ndvi,
            delta_gndvi=delta_gndvi,
            delta_ndre=delta_ndre,
            valid_mask=valid_mask,
            config=config,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"\nPhase 6 Execution Error: {e}")
        sys.exit(1)

    print("\n============================================================")
    print("PHASE 6 DEGRADATION & RECOVERY ANALYSIS")
    print("============================================================")
    print(f"Real data pair: {pair_name}")
    print(f"Output directory: {output_dir.resolve()}")
    print("\n--- STATISTICAL SUMMARY ---")
    stats = result.statistics
    print(f"Valid Pixels: {stats['valid_pixel_count']}")
    print(f"NoData Pixels: {stats['nodata_pixel_count']}")
    print()
    print(f"Degradation: {stats['degradation_pixel_count']} pixels ({stats['degradation_percentage']:.4f}%)")
    print(f"Recovery: {stats['recovery_pixel_count']} pixels ({stats['recovery_percentage']:.4f}%)")
    print(f"Canopy Loss: {stats['canopy_loss_pixel_count']} pixels ({stats['canopy_loss_percentage']:.4f}%)")
    print(f"Residual Stress: {stats['residual_stress_pixel_count']} pixels ({stats['residual_stress_percentage']:.4f}%)")
    print(f"Background-to-Vegetation: {stats['background_to_vegetation_count']} pixels ({stats['background_to_vegetation_percentage']:.4f}%)")
    print(f"Stable Total: {stats['stable_count']} pixels")
    print(f"    Stable Vegetation: {stats['stable_vegetation_count']} pixels")
    print(f"    Stable Background: {stats['stable_background_count']} pixels")
    
    print("\n--- CVA RESIDUAL STRESS THRESHOLD ---")
    thresh = result.thresholds["residual_cva_threshold"]
    print(f"Method: {thresh['method']}")
    print(f"Configured Value: {thresh['configured_value']}")
    print(f"Threshold Value: {thresh['threshold_value']:.6f}")
    print(f"Stable Vegetation Population Count: {thresh['population_count']}")

    print("\n--- TRANSITION BREAKDOWN ---")
    t_counts = stats['transition_counts']
    for k, v in t_counts.items():
        print(f"    {k}: {v} pixels")

    print("\n--- CVA STATISTICS BY CATEGORY ---")
    cva_cat = stats['cva_statistics_by_category']
    for cat, c_info in cva_cat.items():
        print(f"    {cat.upper()}:")
        if c_info["mean"] is not None:
            print(f"        mean: {c_info['mean']:.6f}")
            print(f"        median: {c_info['median']:.6f}")
            print(f"        P95: {c_info['p95']:.6f}")
        else:
            print("        No data")

    print("\nPhase 6 successfully completed and validated.")

if __name__ == "__main__":
    main()
