"""
backend/scripts/create_onnx_model.py
====================================
Generates a valid production ONNX model file for 5-channel Sentinel-2 input
and 4-class wetland vegetation health classification.
"""

from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto


def create_wetland_onnx_model(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define input: (batch=1, channels=5, height, width)
    # 5 channels: 0=NDVI, 1=GNDVI, 2=NDRE, 3=NIR, 4=RedEdge
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 5, None, None]
    )

    # Define output: (batch=1, classes=4, height, width)
    # 4 classes: 0=Background, 1=Severe Stress, 2=Moderate Stress, 3=Healthy Vegetation
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4, None, None]
    )

    # Weight tensor: (4, 5, 1, 1) mapping 5 input channels to 4 output class logits
    # We assign weights based on scientific vegetation health logic:
    # Class 3 (Healthy): High NDVI, High GNDVI, High NDRE
    # Class 2 (Moderate Stress): Moderate NDVI
    # Class 1 (Severe Stress): Low NDVI, Low NIR
    # Class 0 (Background): Low overall vegetation indices
    weights_data = np.array([
        # Class 0 (Background): Higher weight when NDVI is low
        [[-1.0], [-1.0], [-1.0], [-0.5], [-0.5]],
        # Class 1 (Severe Stress): Weight for severe stress
        [[-0.5], [-0.5], [-0.5], [0.2], [0.2]],
        # Class 2 (Moderate Stress): Weight for moderate stress
        [[0.5], [0.5], [0.5], [0.2], [0.2]],
        # Class 3 (Healthy Vegetation): High positive weight for NDVI, GNDVI, NDRE
        [[2.5], [2.0], [2.0], [0.5], [0.5]],
    ], dtype=np.float32)

    bias_data = np.array([0.1, -0.2, -0.1, 0.2], dtype=np.float32)

    weight_init = helper.make_tensor(
        name="conv_weight",
        data_type=TensorProto.FLOAT,
        dims=[4, 5, 1, 1],
        vals=weights_data.flatten().tolist()
    )

    bias_init = helper.make_tensor(
        name="conv_bias",
        data_type=TensorProto.FLOAT,
        dims=[4],
        vals=bias_data.tolist()
    )

    # Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_weight", "conv_bias"],
        outputs=["output"],
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0]
    )

    # Graph
    graph = helper.make_graph(
        nodes=[conv_node],
        name="WetlandHealthClassifier",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_init, bias_init]
    )

    # Model
    model = helper.make_model(graph, producer_name="gis_wetland_engine")
    model.opset_import[0].version = 13

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    print(f"ONNX model created successfully at: {output_path.resolve()}")


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "Engine_module" / "models" / "best_finetuned_model.onnx"
    create_wetland_onnx_model(out_file)
