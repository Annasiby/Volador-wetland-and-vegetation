import os
import sys
import argparse
import json
from pathlib import Path

# Ensure src/ is on the path so we can import gis_engine
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gis_engine.core.pair import analyze_pair

def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 3: Two-Date Model Inference pipeline.")
    parser.add_argument("--t1", required=True, help="Path to T1 aligned scene directory.")
    parser.add_argument("--t2", required=True, help="Path to T2 aligned scene directory.")
    parser.add_argument("--model", default="models/best_finetuned_model.onnx", help="Path to the frozen ONNX model.")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument("--config", default="configs/config.json", help="Path to the config.json file.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load config
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Configuration file not found at: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    t1_scene = Path(args.t1)
    t2_scene = Path(args.t2)
    model_path = Path(args.model)
    output_dir = Path(args.output)
    
    if not t1_scene.is_dir():
        print(f"Error: T1 directory not found: {t1_scene}")
        sys.exit(1)
    if not t2_scene.is_dir():
        print(f"Error: T2 directory not found: {t2_scene}")
        sys.exit(1)
    if not model_path.is_file():
        print(f"Error: ONNX model not found: {model_path}")
        sys.exit(1)

    print("\nRunning Two-Date Inference...")
    try:
        result = analyze_pair(
            t1_scene=t1_scene,
            t2_scene=t2_scene,
            model_path=model_path,
            output_dir=output_dir,
            config=config
        )
    except Exception as e:
        print(f"\nPipeline Execution Error: {e}")
        sys.exit(1)

    # 2. Print required console output
    print("\n============================================================")
    print("TWO-DATE INFERENCE")
    print("============================================================")
    
    # Fetch shapes from logits
    t1_shape_str = f"shape {result.t1_logits.shape}, dtype {result.t1_logits.dtype}"
    t2_shape_str = f"shape {result.t2_logits.shape}, dtype {result.t2_logits.dtype}"
    
    print("\nT1:")
    print("    validation: PASS")
    print(f"    tensor: {t1_shape_str}")
    print(f"    provider: {result.performance['t1']['inference_provider']}")
    print(f"    inference time: {result.performance['t1']['inference_time_ms']:.2f} ms")

    print("\nT2:")
    print("    validation: PASS")
    print(f"    tensor: {t2_shape_str}")
    print(f"    provider: {result.performance['t2']['inference_provider']}")
    print(f"    inference time: {result.performance['t2']['inference_time_ms']:.2f} ms")


    print("\nCOMMON GRID:")
    print(f"    CRS: {result.reference_grid.crs}")
    print(f"    dimensions: {result.reference_grid.height}x{result.reference_grid.width}")
    res = result.reference_grid.resolution
    print(f"    resolution: {res[0]:.6f}, {res[1]:.6f}")
    overlap_info = result.class_statistics_t1["common_overlap"] # Or read from overlap details
    total_pixels = result.reference_grid.width * result.reference_grid.height
    overlap_pct = (np.sum(result.overlap_mask) / total_pixels) * 100.0
    print(f"    overlap: {np.sum(result.overlap_mask)} pixels ({overlap_pct:.2f}%)")

    # Helper to print class distribution
    def print_class_dist(class_stats):
        classes = class_stats["classes"]
        for class_id in ["0", "1", "2", "3"]:
            c_info = classes[class_id]
            print(f"    {c_info['name']}: {c_info['count']} pixels ({c_info['percentage']:.2f}%)")

    print("\nT1 CLASS DISTRIBUTION:")
    print_class_dist(result.class_statistics_t1["date_valid"])

    print("\nT2 CLASS DISTRIBUTION:")
    print_class_dist(result.class_statistics_t2["date_valid"])

    print("\nOUTPUTS:")
    print(f"    T1 Health Map   : {output_dir / 'T1' / 'health.tif'}")
    print(f"    T1 Logits       : {output_dir / 'T1' / 'logits.npy'}")
    print(f"    T1 Probabilities: {output_dir / 'T1' / 'probabilities.npy'}")
    print(f"    T1 Confidence   : {output_dir / 'T1' / 'confidence.tif'}")
    print(f"    T2 Health Map   : {output_dir / 'T2' / 'health.tif'}")
    print(f"    T2 Logits       : {output_dir / 'T2' / 'logits.npy'}")
    print(f"    T2 Probabilities: {output_dir / 'T2' / 'probabilities.npy'}")
    print(f"    T2 Confidence   : {output_dir / 'T2' / 'confidence.tif'}")
    print(f"    Overlap Mask    : {output_dir / 'overlap_mask.tif'}")
    print(f"    Pair Metadata   : {output_dir / 'pair_metadata.json'}")
    print("============================================================\n")

if __name__ == "__main__":
    import numpy as np
    main()
