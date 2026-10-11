import os
import sys
import argparse
import json
from pathlib import Path

# Ensure src/ is on the path so we can import gis_engine
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gis_engine.change.transition import calculate_transitions, TRANSITIONS

def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 4: Post-Classification Transition Engine.")
    parser.add_argument("--t1", required=True, help="Path to T1 health map GeoTIFF.")
    parser.add_argument("--t2", required=True, help="Path to T2 health map GeoTIFF.")
    parser.add_argument("--mask", required=True, help="Path to valid overlap mask GeoTIFF.")
    parser.add_argument("--output", required=True, help="Path to output directory.")
    return parser.parse_args()

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Fallback for python environments where reconfigure is not available
        
    args = parse_args()
    
    t1_path = Path(args.t1)
    t2_path = Path(args.t2)
    mask_path = Path(args.mask)
    output_dir = Path(args.output)
    
    if not t1_path.is_file():
        print(f"Error: T1 health map not found: {t1_path}")
        sys.exit(1)
    if not t2_path.is_file():
        print(f"Error: T2 health map not found: {t2_path}")
        sys.exit(1)
    if not mask_path.is_file():
        print(f"Error: Overlap mask not found: {mask_path}")
        sys.exit(1)
        
    try:
        result = calculate_transitions(
            t1_health_map=t1_path,
            t2_health_map=t2_path,
            valid_overlap_mask=mask_path,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"\nTransition Calculation Error: {e}")
        sys.exit(1)
        
    # Format and print the console output as expected
    print("\n============================================================")
    print("TRANSITION ANALYSIS")
    print("============================================================")
    
    print("T1:")
    print(f"    path: {t1_path}")
    print(f"    dimensions: {result.reference_grid.height}x{result.reference_grid.width}")
    
    print("\nT2:")
    print(f"    path: {t2_path}")
    print(f"    dimensions: {result.reference_grid.height}x{result.reference_grid.width}")
    
    print("\nCOMMON GRID:")
    print(f"    CRS: {result.reference_grid.crs}")
    print(f"    dimensions: {result.reference_grid.height}x{result.reference_grid.width}")
    res = result.reference_grid.resolution
    if res is not None:
        print(f"    resolution: {res[0]:.6f}, {res[1]:.6f}")
    else:
        print("    resolution: Unknown")
        
    total_valid_pixels = sum(result.transition_counts.values())
    total_valid_area_ha = sum(result.transition_areas_ha.values())
    print("\nVALID OVERLAP:")
    print(f"    pixels: {total_valid_pixels}")
    print(f"    area: {total_valid_area_ha:.4f} ha")
    
    print("\nTRANSITIONS:")
    
    # Order: Row-major transitions
    # B->B, B->S, B->M, B->H
    # S->B, S->S, S->M, S->H
    # M->B, M->S, M->M, M->H
    # H->B, H->S, H->M, H->H
    def format_transition_line(tid, label):
        count = result.transition_counts[tid]
        pct = result.transition_percentages[tid]
        area_ha = result.transition_areas_ha[tid]
        return f"{label}: {count} pixels ({pct:.2f}%) - {area_ha:.4f} ha"

    print(format_transition_line(0, "B→B"))
    print(format_transition_line(1, "B→S"))
    print(format_transition_line(2, "B→M"))
    print(format_transition_line(3, "B→H"))
    print()
    print(format_transition_line(4, "S→B"))
    print(format_transition_line(5, "S→S"))
    print(format_transition_line(6, "S→M"))
    print(format_transition_line(7, "S→H"))
    print()
    print(format_transition_line(8, "M→B"))
    print(format_transition_line(9, "M→S"))
    print(format_transition_line(10, "M→M"))
    print(format_transition_line(11, "M→H"))
    print()
    print(format_transition_line(12, "H→B"))
    print(format_transition_line(13, "H→S"))
    print(format_transition_line(14, "H→M"))
    print(format_transition_line(15, "H→H"))
    
    print("\nCATEGORIES:")
    
    def format_category_section(cat, label):
        count = result.category_counts[cat]
        pct = result.category_percentages[cat]
        area_ha = result.category_areas_ha[cat]
        print(f"{label}:")
        print(f"    pixels: {count} ({pct:.2f}%)")
        print(f"    area: {area_ha:.4f} ha")
        
    format_category_section("DEGRADATION", "DEGRADATION")
    print()
    format_category_section("RECOVERY", "RECOVERY")
    print()
    format_category_section("VEGETATION_TO_BACKGROUND", "VEGETATION→BACKGROUND")
    print()
    format_category_section("BACKGROUND_TO_VEGETATION", "BACKGROUND→VEGETATION")
    print()
    format_category_section("STABLE", "STABLE")
    
    print("============================================================\n")

if __name__ == "__main__":
    main()
