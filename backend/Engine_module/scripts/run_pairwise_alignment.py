import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
import rasterio

# Ensure src/ is on the path so we can import gis_engine
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gis_engine.alignment.temporal import TemporalAligner

def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 2: Pairwise Temporal/Spatial Alignment.")
    parser.add_argument("--t1", required=True, help="Path to T1 input scene directory.")
    parser.add_argument("--t1-date", required=True, help="T1 date in YYYY-MM-DD format.")
    parser.add_argument("--t2", required=True, help="Path to T2 input scene directory.")
    parser.add_argument("--t2-date", required=True, help="T2 date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default="outputs/alignment", help="Base directory for output aligned pairs.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config.json file.")
    return parser.parse_args()

def write_band_tiff(output_path: Path, array: np.ndarray, grid, nodata_val):
    """
    Saves an aligned band array as a georeferenced single-band float32 GeoTIFF.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        'driver': 'GTiff',
        'height': grid.height,
        'width': grid.width,
        'count': 1,
        'dtype': rasterio.float32,
        'crs': grid.crs,
        'transform': grid.transform,
        'nodata': nodata_val if nodata_val is not None else np.nan,
        'compress': 'lzw'
    }
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(np.float32), 1)

def write_mask_tiff(output_path: Path, mask: np.ndarray, grid):
    """
    Saves a boolean mask as a georeferenced single-band uint8 GeoTIFF.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        'driver': 'GTiff',
        'height': grid.height,
        'width': grid.width,
        'count': 1,
        'dtype': rasterio.uint8,
        'crs': grid.crs,
        'transform': grid.transform,
        'nodata': 0,
        'compress': 'lzw'
    }
    # Save as 1 for valid, 0 for invalid
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)

def main():
    args = parse_args()
    
    # 1. Load config
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Configuration file not found at: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = json.load(f)

    print("\n" + "=" * 60)
    print("GIS WETLAND DEGRADATION & DEPLETION ENGINE (PHASE 2 - ALIGNMENT)")
    print("=" * 60)
    print(f"T1 Input Directory : {args.t1} (Date: {args.t1_date})")
    print(f"T2 Input Directory : {args.t2} (Date: {args.t2_date})")
    print(f"Base Output Dir    : {args.output_dir}")
    print("=" * 60)

    # 2. Run alignment
    try:
        aligner = TemporalAligner(config)
        pair_name = f"{args.t1_date}_{args.t2_date}"
        output_pair_dir = Path(args.output_dir) / pair_name

        print("\n[1/3] Running Validation & Spatial Alignment...")
        pair = aligner.align_pair(
            t1_dir=Path(args.t1),
            t1_date=args.t1_date,
            t2_dir=Path(args.t2),
            t2_date=args.t2_date
        )
        print("  - Alignment calculation completed.")
        print(f"  - Alignment required: {pair.alignment_required}")
        if pair.alignment_required:
            print(f"  - Operations performed: {pair.alignment_operations}")
        else:
            print("  - Grids are identical. T1 and T2 are already aligned.")

    except Exception as e:
        print(f"\nAlignment Error: {e}")
        sys.exit(1)

    # 3. Write outputs
    print("\n[2/3] Writing Aligned GeoTIFFs & Metadata...")
    t1_aligned_dir = output_pair_dir / "t1_aligned"
    t2_aligned_dir = output_pair_dir / "t2_aligned"
    
    try:
        # Write T1 bands
        t1_nodata = pair.t1_metadata.get("nodata", np.nan)
        for band_name, band_arr in pair.t1_data.items():
            write_band_tiff(t1_aligned_dir / f"{band_name}.tif", band_arr, pair.reference_grid, t1_nodata)
        print(f"  - Saved aligned T1 bands to: {t1_aligned_dir}")

        # Write T2 bands
        t2_nodata = pair.t2_original_metadata.get("nodata", np.nan)
        for band_name, band_arr in pair.t2_data.items():
            write_band_tiff(t2_aligned_dir / f"{band_name}.tif", band_arr, pair.reference_grid, t2_nodata)
        print(f"  - Saved aligned T2 bands to: {t2_aligned_dir}")

        # Write overlap mask
        mask_path = output_pair_dir / "overlap_mask.tif"
        write_mask_tiff(mask_path, pair.overlap_mask, pair.reference_grid)
        print(f"  - Saved valid-overlap mask to: {mask_path}")

        # Write metadata JSON
        metadata_path = output_pair_dir / "alignment_metadata.json"
        
        # Serialize rasterio CRS and transform correctly for JSON
        serializable_metadata = {
            "reference": "T1",
            "t1_original_crs": str(pair.t1_metadata.get("crs")),
            "t2_original_crs": str(pair.t2_original_metadata.get("crs")),
            "t1_resolution": list(pair.t1_metadata.get("resolution")),
            "t2_resolution": list(pair.t2_original_metadata.get("resolution")),
            "t1_shape": [pair.t1_metadata.get("height"), pair.t1_metadata.get("width")],
            "t2_shape": [pair.t2_original_metadata.get("height"), pair.t2_original_metadata.get("width")],
            "alignment_required": pair.alignment_required,
            "alignment_operations": pair.alignment_operations,
            "t2_reprojected": "reproject_crs" in pair.alignment_operations,
            "t2_resampling": aligner.resampling_method,
            "target_width": pair.reference_grid.width,
            "target_height": pair.reference_grid.height,
            "target_resolution": list(pair.reference_grid.resolution),
            "target_transform": [
                pair.reference_grid.transform.a, pair.reference_grid.transform.b, pair.reference_grid.transform.c,
                pair.reference_grid.transform.d, pair.reference_grid.transform.e, pair.reference_grid.transform.f
            ],
            "spatial_overlap": True,
            "overlap_percentage": float(pair.overlap_percentage)
        }
        
        with open(metadata_path, "w") as f:
            json.dump(serializable_metadata, f, indent=4)
        print(f"  - Saved alignment metadata to: {metadata_path}")

    except Exception as e:
        print(f"\nError writing output files: {e}")
        sys.exit(1)

    # 4. Report final results
    print("\n[3/3] Spatial Alignment Report Summary")
    print("=" * 60)
    
    # Format bounds function
    def fmt_bounds(b):
        return f"L:{b[0]:.6f}, B:{b[1]:.6f}, R:{b[2]:.6f}, T:{b[3]:.6f}"

    print("T1 original:")
    print(f"    CRS        : {pair.t1_metadata.get('crs')}")
    print(f"    Resolution : {pair.t1_metadata.get('resolution')}")
    print(f"    Dimensions : [{pair.t1_metadata.get('height')}, {pair.t1_metadata.get('width')}]")
    print(f"    Bounds     : {fmt_bounds(pair.t1_metadata.get('extent'))}")
    print(f"    Transform  : {list(serializable_metadata['target_transform'])}")

    print("\nT2 original:")
    print(f"    CRS        : {pair.t2_original_metadata.get('crs')}")
    print(f"    Resolution : {pair.t2_original_metadata.get('resolution')}")
    print(f"    Dimensions : [{pair.t2_original_metadata.get('height')}, {pair.t2_original_metadata.get('width')}]")
    print(f"    Bounds     : {fmt_bounds(pair.t2_original_metadata.get('extent'))}")
    
    t2_trans_list = [
        pair.t2_original_metadata.get('transform').a, pair.t2_original_metadata.get('transform').b, pair.t2_original_metadata.get('transform').c,
        pair.t2_original_metadata.get('transform').d, pair.t2_original_metadata.get('transform').e, pair.t2_original_metadata.get('transform').f
    ] if pair.t2_original_metadata.get('transform') is not None else None
    print(f"    Transform  : {t2_trans_list}")

    print("\nCommon grid (T1-authoritative):")
    print(f"    CRS        : {serializable_metadata['t1_original_crs']}")
    print(f"    Resolution : {serializable_metadata['target_resolution']}")
    print(f"    Dimensions : [{serializable_metadata['target_height']}, {serializable_metadata['target_width']}]")
    print(f"    Bounds     : {fmt_bounds(pair.reference_grid.bounds)}")
    print(f"    Transform  : {list(serializable_metadata['target_transform'])}")

    print("\nOverlap info:")
    print(f"    Overlap %  : {pair.overlap_percentage:.2f}%")
    print(f"    Alignment  : {'Required' if pair.alignment_required else 'Not Required'}")
    print(f"    Resampling : {aligner.resampling_method}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
