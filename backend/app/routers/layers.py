"""
backend/app/routers/layers.py
==============================
Router for retrieving map layer catalog and rendering GeoTIFF layers as PNG tiles.
"""

from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, Response
from ..schemas import MapLayerInfo
from ..services import RunWorker, TileRenderer

router = APIRouter(prefix="/api/runs/{run_id}/layers", tags=["Layers"])

LAYER_DEFINITIONS = [
    {"id": "t1_health", "name": "T1 — Baseline Health", "type": "raster", "subpath": "vegetation/t1_health.tif", "layer_type": "health"},
    {"id": "t2_health", "name": "T2 — Comparison Health", "type": "raster", "subpath": "vegetation/t2_health.tif", "layer_type": "health"},
    {"id": "transition_map", "name": "4x4 Transition Analysis", "type": "raster", "subpath": "change/transition_map.tif", "layer_type": "change"},
    {"id": "cva_magnitude", "name": "CVA Change Magnitude", "type": "raster", "subpath": "change/cva_magnitude.tif", "layer_type": "cva"},
    {"id": "degradation_map", "name": "Vegetation Degradation", "type": "raster", "subpath": "degradation/degradation_map.tif", "layer_type": "degradation"},
    {"id": "recovery_map", "name": "Vegetation Recovery", "type": "raster", "subpath": "degradation/recovery_map.tif", "layer_type": "recovery"},
    {"id": "water_change", "name": "Surface Water Change", "type": "raster", "subpath": "hydrology/water_change.tif", "layer_type": "water"},
]


@router.get("", response_model=List[MapLayerInfo])
def get_layer_catalog(run_id: str):
    run_info = RunWorker.get_run_status(run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    out_dir = Path(run_info["output_directory"])
    if (out_dir / run_id).is_dir():
        out_dir = out_dir / run_id

    layers = []

    for defn in LAYER_DEFINITIONS:
        file_path = out_dir / defn["subpath"]
        ready = file_path.is_file()
        bounds = None

        if ready:
            try:
                bounds = TileRenderer.get_raster_bounds_wgs84(file_path)
            except Exception:
                bounds = None

        layers.append(MapLayerInfo(
            id=defn["id"],
            name=defn["name"],
            type=defn["type"],
            url=f"/api/runs/{run_id}/layers/{defn['id']}/image",
            format="png",
            bounds=bounds,
            crs="EPSG:4326",
            opacity=0.8,
            ready=ready
        ))

    return layers


@router.get("/{layer_id}/image")
def render_layer_image(run_id: str, layer_id: str):
    run_info = RunWorker.get_run_status(run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    out_dir = Path(run_info["output_directory"])
    if (out_dir / run_id).is_dir():
        out_dir = out_dir / run_id

    defn = next((d for d in LAYER_DEFINITIONS if d["id"] == layer_id), None)
    if not defn:
        raise HTTPException(status_code=404, detail=f"Layer {layer_id} unknown")

    raster_path = out_dir / defn["subpath"]
    if not raster_path.exists():
        raise HTTPException(status_code=404, detail=f"Raster file for layer {layer_id} not generated yet")

    png_bytes = TileRenderer.render_raster_as_png(raster_path, layer_type=defn["layer_type"])
    return Response(content=png_bytes, media_type="image/png")
