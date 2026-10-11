"""
backend/scripts/generate_sample_scene.py
=========================================
Generates realistic georeferenced Sentinel-2 T1 and T2 scenes
along with a wetland boundary vector GeoJSON file for end-to-end testing
and live execution.
"""

import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin


def generate_sample_dataset(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    t1_dir = output_dir / "t1"
    t2_dir = output_dir / "t2"
    t1_dir.mkdir(exist_ok=True)
    t2_dir.mkdir(exist_ok=True)

    # 100x100 pixels at 10m resolution (1 km x 1 km) in UTM Zone 33N
    shape = (100, 100)
    crs = "EPSG:32633"
    # Origin near a wetland area (e.g. E: 500000, N: 5000000)
    transform = from_origin(500000.0, 5000000.0, 10.0, 10.0)

    # Create spatial coordinate grids for features
    rows, cols = np.indices(shape)
    center_y, center_x = 50, 50
    dist_from_center = np.sqrt((rows - center_y)**2 + (cols - center_x)**2)

    # Wetland core (center circle) vs surrounding land
    wetland_mask = dist_from_center <= 30

    # T1 Spectral values (reflectance x 10000)
    # Healthy vegetation: NIR high (~4500), Red low (~800), Green (~1200), RedEdge (~2500)
    # Water/Mud in core: NIR low (~800), Red low (~600), Green (~1000), RedEdge (~900)
    
    t1_green = np.where(wetland_mask, 1000, 1200).astype(np.float32)
    t1_red = np.where(wetland_mask, 600, 800).astype(np.float32)
    t1_rededge = np.where(wetland_mask, 1100, 2600).astype(np.float32)
    t1_nir = np.where(wetland_mask, 1200, 4800).astype(np.float32)

    # Add slight random spatial noise for realism
    rng = np.random.default_rng(42)
    t1_green += rng.normal(0, 30, shape)
    t1_red += rng.normal(0, 30, shape)
    t1_rededge += rng.normal(0, 50, shape)
    t1_nir += rng.normal(0, 80, shape)

    # T2 Spectral values (T2 shows degradation in North-East quadrant of wetland)
    degraded_mask = wetland_mask & (rows < 50) & (cols > 50)
    
    t2_green = t1_green.copy()
    t2_red = t1_red.copy()
    t2_rededge = t1_rededge.copy()
    t2_nir = t1_nir.copy()

    # Degraded area: NIR drops significantly, Red increases (stress)
    t2_nir[degraded_mask] *= 0.45
    t2_red[degraded_mask] *= 1.8
    t2_rededge[degraded_mask] *= 0.6

    bands_t1 = {
        "scene_G.tif": t1_green,
        "scene_R.tif": t1_red,
        "scene_RE.tif": t1_rededge,
        "scene_NIR.tif": t1_nir,
    }

    bands_t2 = {
        "scene_G.tif": t2_green,
        "scene_R.tif": t2_red,
        "scene_RE.tif": t2_rededge,
        "scene_NIR.tif": t2_nir,
    }

    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
    }

    for filename, data in bands_t1.items():
        with rasterio.open(t1_dir / filename, "w", **profile) as dst:
            dst.write(data, 1)

    for filename, data in bands_t2.items():
        with rasterio.open(t2_dir / filename, "w", **profile) as dst:
            dst.write(data, 1)

    # Create GeoJSON Wetland Boundary matching the spatial extent (UTM Zone 33N coordinates)
    min_x, max_y = 500000.0, 5000000.0
    max_x, min_y = 500000.0 + (100 * 10.0), 5000000.0 - (100 * 10.0)

    # A polygon encompassing the center wetland zone
    poly_min_x = min_x + 200.0
    poly_max_x = max_x - 200.0
    poly_min_y = min_y + 200.0
    poly_max_y = max_y - 200.0

    boundary_geojson = {
        "type": "FeatureCollection",
        "name": "Wetland_Boundary",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::32633"}
        },
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1, "name": "Central Wetland Reserve"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [poly_min_x, poly_min_y],
                        [poly_max_x, poly_min_y],
                        [poly_max_x, poly_max_y],
                        [poly_min_x, poly_max_y],
                        [poly_min_x, poly_min_y]
                    ]]
                }
            }
        ]
    }

    boundary_path = output_dir / "wetland_boundary.geojson"
    with open(boundary_path, "w") as f:
        json.dump(boundary_geojson, f, indent=2)

    print(f"Sample dataset generated successfully in: {output_dir.resolve()}")
    return output_dir


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "Engine_module" / "data" / "sample_scene"
    generate_sample_dataset(out)
