import json
import logging
import shapefile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from rasterio.crs import CRS
from rasterio.warp import transform

logger = logging.getLogger(__name__)

def reproject_geometry(geom: Dict[str, Any], src_crs: CRS, dst_crs: CRS) -> Dict[str, Any]:
    """
    Reprojects GeoJSON-like geometry coordinates from src_crs to dst_crs.
    Supports Polygon and MultiPolygon geometry types.
    """
    if src_crs == dst_crs:
        return geom

    g_type = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return geom

    if g_type == "Polygon":
        new_coords = []
        for ring in coords:
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            new_xs, new_ys = transform(src_crs, dst_crs, xs, ys)
            new_ring = list(zip(new_xs, new_ys))
            new_coords.append(new_ring)
        return {"type": "Polygon", "coordinates": new_coords}

    elif g_type == "MultiPolygon":
        new_coords = []
        for poly in coords:
            new_poly = []
            for ring in poly:
                xs = [pt[0] for pt in ring]
                ys = [pt[1] for pt in ring]
                new_xs, new_ys = transform(src_crs, dst_crs, xs, ys)
                new_ring = list(zip(new_xs, new_ys))
                new_poly.append(new_ring)
            new_coords.append(new_poly)
        return {"type": "MultiPolygon", "coordinates": new_coords}

    else:
        # Fallback or pass-through for other shapes
        return geom

def load_vector_boundary(
    file_path: Path,
    class_field: Optional[str] = None,
    select_classes: Optional[List[Any]] = None
) -> Tuple[List[Dict[str, Any]], CRS]:
    """
    Loads vector boundaries from Shapefile or GeoJSON.
    Filters by specific classes if class_field and select_classes are provided.
    
    Returns:
        List of GeoJSON-like geometry dictionaries and the source CRS.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Boundary file not found: {file_path}")

    suffix = file_path.suffix.lower()
    
    if suffix in (".geojson", ".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        crs = None
        if "crs" in data:
            try:
                crs_properties = data["crs"].get("properties", {})
                crs_name = crs_properties.get("name")
                if crs_name:
                    crs = CRS.from_string(crs_name)
            except Exception as exc:
                logger.warning(
                    "[boundary] Could not parse CRS from GeoJSON 'crs' member "
                    "(file: %s): %s. Defaulting to EPSG:4326.",
                    file_path.name, exc
                )
        if crs is None:
            crs = CRS.from_epsg(4326) # Default to EPSG:4326 for GeoJSON
            
        features = []
        def parse_element(element):
            if element.get("type") == "FeatureCollection":
                for feat in element.get("features", []):
                    parse_element(feat)
            elif element.get("type") == "Feature":
                geom = element.get("geometry")
                props = element.get("properties", {}) or {}
                if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
                    if class_field and select_classes:
                        val = props.get(class_field)
                        if val not in select_classes:
                            return
                    features.append(geom)
            elif element.get("type") in ("Polygon", "MultiPolygon"):
                features.append(element)
                
        parse_element(data)
        return features, crs

    elif suffix in (".shp", ""):
        # pyshp can read shapefile given path to .shp or base path
        sf = shapefile.Reader(str(file_path))
        
        # Load CRS from .prj if it exists
        crs = None
        prj_path = file_path.with_suffix(".prj")
        if prj_path.is_file():
            try:
                with open(prj_path, "r", encoding="utf-8") as f:
                    wkt = f.read().strip()
                if wkt:
                    crs = CRS.from_string(wkt)
            except Exception as exc:
                logger.warning(
                    "[boundary] Could not parse CRS from .prj file "
                    "(file: %s): %s. Defaulting to EPSG:4326.",
                    prj_path.name, exc
                )
        if crs is None:
            crs = CRS.from_epsg(4326)
            
        features = []
        fields = [f[0] for f in sf.fields[1:]] # strip deletion flag
        for shape_rec in sf.iterShapeRecords():
            geom = shape_rec.shape.__geo_interface__
            rec = shape_rec.record
            properties = dict(zip(fields, rec))
            
            if geom.get("type") in ("Polygon", "MultiPolygon"):
                if class_field and select_classes:
                    val = properties.get(class_field)
                    # Handle byte strings in shapefiles
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="ignore")
                    # Make sure selection mapping is robust to types
                    str_select = [str(c) for c in select_classes]
                    if str(val) not in str_select:
                        continue
                features.append(geom)
                
        return features, crs

    else:
        raise ValueError(f"Unsupported boundary format: {suffix}")
