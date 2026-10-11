import os
from pathlib import Path
import rasterio
from rasterio.transform import from_origin

def main():
    t1_src_dir = Path("data/potato sample data/20230814123320_0001")
    t2_src_dir = Path("data/potato sample data/20230828111806_0002")
    
    t1_dst_dir = Path("data/real_georeferenced/2023-08-14")
    t2_dst_dir = Path("data/real_georeferenced/2023-08-28")
    
    t1_dst_dir.mkdir(parents=True, exist_ok=True)
    t2_dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Georeference T1 (2023-08-14)
    # Using EPSG:4326, pixel size 0.00001 degrees (~1.1 meter)
    t1_transform = from_origin(100.0, 10.0, 0.00001, 0.00001)
    
    # 2. Georeference T2 (2023-08-28)
    # Shifted slightly by 0.00005 degrees (5 pixels) to the east/north and slightly different resolution (0.000012)
    t2_transform = from_origin(100.00005, 10.00005, 0.000012, 0.000012)
    
    print("Preparing georeferenced datasets...")
    
    # Georeference T1 bands
    for path in t1_src_dir.glob("*.TIF"):
        band_name = path.name.split("_")[-1]  # G.TIF, NIR.TIF, R.TIF, RE.TIF
        dst_path = t1_dst_dir / f"scene_{band_name.lower()}"
        with rasterio.open(path) as src:
            data = src.read(1)
            profile = src.profile.copy()
            profile.update({
                'crs': 'EPSG:4326',
                'transform': t1_transform,
                'nodata': 0 # Assign 0 as nodata for demo purposes
            })
            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(data, 1)
    print(f"Saved georeferenced T1 to: {t1_dst_dir}")
    
    # Georeference T2 bands
    for path in t2_src_dir.glob("*.TIF"):
        band_name = path.name.split("_")[-1]
        dst_path = t2_dst_dir / f"scene_{band_name.lower()}"
        with rasterio.open(path) as src:
            data = src.read(1)
            profile = src.profile.copy()
            profile.update({
                'crs': 'EPSG:4326',
                'transform': t2_transform,
                'nodata': 0
            })
            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(data, 1)
    print(f"Saved georeferenced T2 to: {t2_dst_dir}")

if __name__ == "__main__":
    main()
