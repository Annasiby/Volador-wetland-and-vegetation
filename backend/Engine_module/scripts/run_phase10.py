import os
import sys
import json
from pathlib import Path

# Ensure workspace root is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gis_engine.wetland.water import analyze_hydrology

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    print("=== GIS Wetland Degradation Engine - Phase 10 Hydrological Evidence Track ===")

    # 1. Load config
    config_path = Path("configs/config.json")
    if not config_path.is_file():
        print(f"Error: Configuration file not found at: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. Setup paths for real data
    pair_name = "2023-08-14_2023-08-28"
    alignment_dir = Path("outputs/alignment") / pair_name
    t1_aligned_dir = alignment_dir / "t1_aligned"
    t2_aligned_dir = alignment_dir / "t2_aligned"
    overlap_mask_path = alignment_dir / "overlap_mask.tif"
    
    wetland_mask_path = Path("outputs/wetland") / "wetland_mask.tif"
    output_dir = Path("outputs/hydrology")

    # Verify aligned inputs exist
    for p in [t1_aligned_dir, t2_aligned_dir, overlap_mask_path]:
        if not p.exists():
            print(f"Error: Required aligned input does not exist: {p}")
            print("Please run Phase 2 spatial/temporal alignment first.")
            sys.exit(1)

    print(f"Ingesting aligned scenes for pair: {pair_name}")
    
    # Check if wetland mask exists
    wetland_mask_arg = wetland_mask_path if wetland_mask_path.is_file() else None
    if wetland_mask_arg:
        print("Authoritative wetland mask from Phase 7 detected and will be used to calculate restricted metrics.")
    else:
        print("No authoritative wetland mask detected. Restricted metrics calculation will be skipped.")

    # 3. Execute Phase 10 analysis
    try:
        metrics = analyze_hydrology(
            t1_aligned_dir=t1_aligned_dir,
            t2_aligned_dir=t2_aligned_dir,
            output_dir=output_dir,
            config=config,
            wetland_mask_path=wetland_mask_arg,
            overlap_mask_path=overlap_mask_path
        )
    except Exception as e:
        print(f"Error during execution of Phase 10: {e}")
        sys.exit(1)

    # 4. Print structured summary report
    print("\n============================================================")
    print("PHASE 10 HYDROLOGICAL EVIDENCE REPORT")
    print("============================================================")
    print(f"Real data pair       : {pair_name}")
    print(f"NDWI Index           : {metrics['index']}")
    print(f"Provisional Threshold: {metrics['threshold']:.2f}")
    print(f"Valid Pixel Count    : {metrics['valid_pixel_count']}")
    print(f"NoData Pixel Count   : {metrics['nodata_pixel_count']}")
    print(f"Output Directory     : {output_dir.resolve()}")
    
    print("\n--- SURFACE-WATER EXTENT CHANGE (ALL VALID OVERLAP) ---")
    print(f"Water Area T1        : {metrics['water_area_t1_ha']:.4f} ha ({metrics['water_area_t1_m2']:.2f} m²)")
    print(f"Water Area T2        : {metrics['water_area_t2_ha']:.4f} ha ({metrics['water_area_t2_m2']:.2f} m²)")
    print(f"Water Gain Area      : {metrics['water_gain_ha']:.4f} ha ({metrics['water_gain_m2']:.2f} m²)")
    print(f"Water Loss Area      : {metrics['water_loss_ha']:.4f} ha ({metrics['water_loss_m2']:.2f} m²)")
    print(f"Net Water Change     : {metrics['net_water_change_ha']:.4f} ha ({metrics['net_water_change_m2']:.2f} m²)")

    if metrics["wetland_boundary_used"] and "wetland_restricted_metrics" in metrics:
        wm = metrics["wetland_restricted_metrics"]
        print("\n--- WETLAND-RESTRICTED SURFACE-WATER EXTENT CHANGE ---")
        print(f"Water Area T1        : {wm['water_area_t1_ha']:.4f} ha ({wm['water_area_t1_m2']:.2f} m²)")
        print(f"Water Area T2        : {wm['water_area_t2_ha']:.4f} ha ({wm['water_area_t2_m2']:.2f} m²)")
        print(f"Water Gain Area      : {wm['water_gain_ha']:.4f} ha ({wm['water_gain_m2']:.2f} m²)")
        print(f"Water Loss Area      : {wm['water_loss_ha']:.4f} ha ({wm['water_loss_m2']:.2f} m²)")
        print(f"Net Water Change     : {wm['net_water_change_ha']:.4f} ha ({wm['net_water_change_m2']:.2f} m²)")

    print("\n--- GENERATED FILES ---")
    print(f"T1 NDWI Map          : {output_dir / 'ndwi_t1.tif'}")
    print(f"T2 NDWI Map          : {output_dir / 'ndwi_t2.tif'}")
    print(f"Delta NDWI Map       : {output_dir / 'delta_ndwi.tif'}")
    print(f"T1 Water Mask        : {output_dir / 'water_mask_t1.tif'}")
    print(f"T2 Water Mask        : {output_dir / 'water_mask_t2.tif'}")
    print(f"Water Change Map     : {output_dir / 'water_change.tif'}")
    print(f"Water Metrics JSON   : {output_dir / 'water_metrics.json'}")
    print("============================================================\n")

    print("Phase 10 execution completed successfully.")

if __name__ == "__main__":
    main()
