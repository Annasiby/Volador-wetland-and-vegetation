#!/usr/bin/env python3
"""
scripts/run_analysis.py
==========================
Production CLI for the GIS Wetland Degradation & Depletion Engine.

Usage
-----
    python scripts/run_analysis.py \\
        --t1    <T1_SCENE_DIR>      \\
        --t2    <T2_SCENE_DIR>      \\
        --t1-date YYYY-MM-DD        \\
        --t2-date YYYY-MM-DD        \\
        --config  configs/config.json \\
        --model   models/best_finetuned_model.onnx \\
        --output  outputs/runs       \\
        [--wetland-boundary <PATH>]  \\
        [--run-id   <RUN_ID>]        \\
        [--log-level INFO]

Exit Codes
----------
    0  — SUCCESS or WARNING (processing complete, outputs written)
    1  — Validation / input error (InputValidationError)
    2  — Configuration error (ConfigurationError)
    3  — Alignment error (AlignmentError)
    4  — Inference error (InferenceError)
    5  — Output / IO error (OutputError)
    6  — Internal / unexpected error
"""

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_analysis.py",
        description=(
            "GIS Wetland Degradation & Depletion Engine V1 — "
            "Analyses vegetation health and surface-water change between two dates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Required ---
    req = p.add_argument_group("Required arguments")
    req.add_argument(
        "--t1", required=True, metavar="PATH",
        help="Path to T1 (baseline) scene directory.",
    )
    req.add_argument(
        "--t2", required=True, metavar="PATH",
        help="Path to T2 (target) scene directory.",
    )
    req.add_argument(
        "--t1-date", required=True, metavar="YYYY-MM-DD",
        help="T1 acquisition date in YYYY-MM-DD format.",
    )
    req.add_argument(
        "--t2-date", required=True, metavar="YYYY-MM-DD",
        help="T2 acquisition date in YYYY-MM-DD format. Must be later than --t1-date.",
    )
    req.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to the JSON configuration file (e.g. configs/config.json).",
    )
    req.add_argument(
        "--model", required=True, metavar="PATH",
        help="Path to the ONNX model file (e.g. models/best_finetuned_model.onnx).",
    )
    req.add_argument(
        "--output", required=True, metavar="PATH",
        help="Base output directory. A run sub-directory is created automatically.",
    )

    # --- Optional ---
    opt = p.add_argument_group("Optional arguments")
    opt.add_argument(
        "--wetland-boundary", default=None, metavar="PATH",
        help="Path to a wetland boundary vector file (.geojson or .shp). "
             "When provided, wetland-restricted metrics are included.",
    )
    opt.add_argument(
        "--run-id", default=None, metavar="ID",
        help="Override the auto-generated run ID (must be unique within --output).",
    )
    opt.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO.",
    )
    opt.add_argument(
        "--json-summary", action="store_true",
        help="If set, print a compact JSON summary of the result to stdout on completion.",
    )
    return p


def _error_code_for(exc_class_name: str) -> int:
    """Maps exception class names to CLI exit codes."""
    mapping = {
        "InputValidationError": 1,
        "ConfigurationError": 2,
        "AlignmentError": 3,
        "InferenceError": 4,
        "OutputError": 5,
    }
    return mapping.get(exc_class_name, 6)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve paths
    t1_scene         = Path(args.t1)
    t2_scene         = Path(args.t2)
    config_path      = Path(args.config)
    model_path       = Path(args.model)
    output_dir       = Path(args.output)
    wetland_boundary = Path(args.wetland_boundary) if args.wetland_boundary else None

    # Quick pre-flight: don't import heavy modules before arg validation
    if not t1_scene.is_dir():
        print(f"[ERROR] T1 scene directory not found: {t1_scene}", file=sys.stderr)
        return 1
    if not t2_scene.is_dir():
        print(f"[ERROR] T2 scene directory not found: {t2_scene}", file=sys.stderr)
        return 1
    if not config_path.is_file():
        print(f"[ERROR] Configuration file not found: {config_path}", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        return 1
    if wetland_boundary and not wetland_boundary.is_file():
        print(f"[WARNING] Wetland boundary file not found: {wetland_boundary}. "
              "Proceeding without wetland context.", file=sys.stderr)
        wetland_boundary = None

    # Add project src to path if running script directly
    src_root = Path(__file__).resolve().parents[1] / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    # Import after path setup
    try:
        from gis_engine.api.service import analyze_wetland
        from gis_engine.core.exceptions import (
            InputValidationError,
            ConfigurationError,
            AlignmentError,
            InferenceError,
            OutputError,
        )
    except ImportError as exc:
        print(f"[ERROR] Failed to import engine modules: {exc}", file=sys.stderr)
        return 6

    print(f"[INFO]  GIS Wetland Engine V1 — run starting")
    print(f"[INFO]  T1: {t1_scene}  ({args.t1_date})")
    print(f"[INFO]  T2: {t2_scene}  ({args.t2_date})")
    print(f"[INFO]  Config: {config_path}")
    print(f"[INFO]  Model:  {model_path}")
    print(f"[INFO]  Output: {output_dir}")
    if wetland_boundary:
        print(f"[INFO]  Wetland boundary: {wetland_boundary}")

    result = analyze_wetland(
        t1_scene=t1_scene,
        t2_scene=t2_scene,
        t1_date=args.t1_date,
        t2_date=args.t2_date,
        config_path=config_path,
        model_path=model_path,
        output_dir=output_dir,
        wetland_boundary=wetland_boundary,
        run_id=args.run_id,
        log_level=args.log_level,
    )

    # Print result summary to console
    print(f"\n[RESULT] run_id     : {result.run_id}")
    print(f"[RESULT] status     : {result.status}")
    print(f"[RESULT] output_dir : {result.output_directory}")

    if result.warnings:
        print(f"[RESULT] warnings   : {len(result.warnings)}")
        for w in result.warnings:
            print(f"           ⚠  {w}")

    if result.errors:
        print(f"[RESULT] errors     : {len(result.errors)}")
        for e in result.errors:
            print(f"           ✖  {e}", file=sys.stderr)

    if args.json_summary:
        summary = {
            "run_id": result.run_id,
            "status": result.status,
            "output_directory": result.output_directory,
            "warnings": result.warnings,
            "errors": result.errors,
            "metrics": result.metrics,
        }
        print(json.dumps(summary, indent=2, default=str))

    # Determine exit code
    if result.status == "FAILED":
        # Try to determine specific exit code from error type
        err_str = " ".join(result.errors)
        for exc_name, code in [
            ("InputValidationError", 1), ("ConfigurationError", 2),
            ("AlignmentError", 3), ("InferenceError", 4), ("OutputError", 5),
        ]:
            if exc_name in err_str:
                return code
        return 6

    return 0  # SUCCESS or WARNING — outputs were written


if __name__ == "__main__":
    sys.exit(main())
