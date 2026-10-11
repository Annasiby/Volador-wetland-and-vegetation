import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

# Ensure src/ is on the path so we can import gis_engine
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gis_engine.inference.model import SingleDateInferenceEngine
from src.gis_engine.output.geotiff import GeoTIFFWriter

CLASS_NAMES = {
    0: "Bare Soil / Background",
    1: "Severe Stress",
    2: "Moderate Stress",
    3: "Healthy Vegetation"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run single-date vegetation health inference pipeline.")
    parser.add_argument("--input", required=True, help="Path to the input scene directory.")
    parser.add_argument("--model", help="Path to the frozen ONNX model.")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument("--config", default="configs/config.json", help="Path to the config.json file.")
    parser.add_argument("--mode", choices=["training", "inference"], help="Override preprocessing mode.")
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
        
    # Override config with CLI arguments
    if args.model:
        config["model"]["path"] = args.model
    if args.mode:
        config["preprocessing"]["mode"] = args.mode

    print("\n" + "=" * 60)
    print("GIS WETLAND DEGRADATION INFERENCE ENGINE (PHASE 1)")
    print("=" * 60)
    print(f"Input Directory   : {args.input}")
    print(f"Model Path        : {config['model']['path']}")
    print(f"Output Directory  : {args.output}")
    print(f"Preprocessing Mode: {config['preprocessing']['mode']}")
    print("=" * 60)

    # Initialize single date inference engine
    try:
        engine = SingleDateInferenceEngine(config)
    except Exception as e:
        print(f"\nInitialization Error: {e}")
        sys.exit(1)

    # Run the pipeline
    try:
        print("\n[1/3] Running Validation & Preprocessing...")
        logits, class_map, metadata, run_stats = engine.run_pipeline(Path(args.input))
        print("  - Input validation passed.")
        print(f"  - Raster dimensions: {metadata.height}x{metadata.width}")
        print(f"  - CRS: {metadata.crs}")
        print(f"  - Preprocessing completed.")
        print(f"  - Tensor created successfully.")
    except Exception as e:
        print(f"\nPipeline Error during Validation/Preprocessing: {e}")
        sys.exit(1)

    # 3. Model execution reporting
    print("\n[2/3] Executing Model Inference...")
    print(f"  - Model input name: {engine.model.input_name}")
    print(f"  - Model output name: {engine.model.output_name}")
    print(f"  - Active execution provider: {run_stats['execution_provider']}")
    print(f"  - Inference execution time: {run_stats['inference_time_ms']:.2f} ms")
    
    # 4. Save results
    print("\n[3/3] Saving Output Files...")
    writer = GeoTIFFWriter(config["output"]["colormap"])
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    health_map_path = output_dir / "health_map.tif"
    logits_path = output_dir / "logits.npy"
    metadata_path = output_dir / "metadata.json"
    
    try:
        writer.write_health_map(str(health_map_path), class_map, metadata)
        writer.write_logits(str(logits_path), logits)
        
        # Determine model input/output shapes for metadata
        H, W = class_map.shape
        input_shape = (1, 5, H, W)
        output_shape = (1, 4, H, W)
        
        writer.write_metadata_json(
            output_path=str(metadata_path),
            input_scene_dir=args.input,
            model_path=config["model"]["path"],
            input_shape=input_shape,
            output_shape=output_shape,
            class_mapping=CLASS_NAMES,
            channel_order=config["model"]["channel_order"],
            preproc_config=config["preprocessing"],
            metadata=metadata,
            inference_provider=run_stats["execution_provider"],
            inference_time_ms=run_stats["inference_time_ms"]
        )
        print(f"  - Saved health map to: {health_map_path}")
        print(f"  - Saved logits to: {logits_path}")
        print(f"  - Saved metadata to: {metadata_path}")
    except Exception as e:
        print(f"\nError saving output files: {e}")
        sys.exit(1)

    # 5. Calculate class distributions
    total_pixels = class_map.size
    print("\n" + "=" * 60)
    print("VEGETATION HEALTH CLASS DISTRIBUTION SANITY CHECK")
    print("=" * 60)
    print(f"Total Pixels: {total_pixels}")
    
    for class_id in [0, 1, 2, 3]:
        count = np.sum(class_map == class_id)
        pct = (count / total_pixels) * 100.0
        print(f"  Class {class_id} ({CLASS_NAMES[class_id]}): {count} pixels ({pct:.2f}%)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
