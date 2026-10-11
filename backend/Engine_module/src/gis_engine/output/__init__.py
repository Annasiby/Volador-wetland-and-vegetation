"""
src/gis_engine/output/__init__.py
==================================
Output generation and reporting package.
"""

from .cartography import (
    generate_report_maps,
    render_categorical_map,
    render_continuous_map,
)
from .geotiff import GeoTIFFWriter
from .json import write_pair_metadata_json
from .pdf import generate_pdf_report
from .report import generate_analysis_report

__all__ = [
    "GeoTIFFWriter",
    "write_pair_metadata_json",
    "generate_analysis_report",
    "generate_pdf_report",
    "generate_report_maps",
    "render_categorical_map",
    "render_continuous_map",
]
