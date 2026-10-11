from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp
import onnx
import onnxruntime as ort


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    r"C:\Users\HP\Desktop\gis_vegetation\5-channel scripts\best_finetuned_model.pth"
)

TEST_TENSOR_PATH = Path(
    r".\outputs\standalone_model_inference\explicit_input\input_tensor.npy"
)

OUTPUT_PATH = Path(
    r".\models\best_finetuned_model_resnet34_test.onnx"
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cpu")

IN_CHANNELS = 5
NUM_CLASSES = 4

OPSET_VERSION = 13


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 70)
print("REAL TRAINED MODEL -> ONNX EXPORT")
print("=" * 70)

print("\n[1] Loading trained checkpoint:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained checkpoint not found: {MODEL_PATH}"
    )

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict):
    state_dict = checkpoint.get("state_dict", checkpoint)
else:
    state_dict = checkpoint

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

if missing:
    raise RuntimeError(
        f"Missing checkpoint parameters: {missing}"
    )

if unexpected:
    raise RuntimeError(
        f"Unexpected checkpoint parameters: {unexpected}"
    )

model.to(DEVICE)
model.eval()

print("[OK] Checkpoint loaded.")
print("[OK] Missing parameters: []")
print("[OK] Unexpected parameters: []")


# ============================================================
# LOAD KNOWN-GOOD INPUT TENSOR
# ============================================================

print("\n[2] Loading known-good model input:")
print(TEST_TENSOR_PATH)

if not TEST_TENSOR_PATH.exists():
    raise FileNotFoundError(
        f"Test tensor not found: {TEST_TENSOR_PATH}"
    )

input_np = np.load(TEST_TENSOR_PATH)

if input_np.dtype != np.float32:
    input_np = input_np.astype(np.float32)

if input_np.ndim != 4:
    raise ValueError(
        f"Expected 4D tensor, got {input_np.shape}"
    )

if input_np.shape[1] != IN_CHANNELS:
    raise ValueError(
        f"Expected 5 channels, got {input_np.shape}"
    )

input_tensor = torch.from_numpy(input_np).to(DEVICE)

print("[OK] Input shape:", input_np.shape)
print("[OK] Input dtype:", input_np.dtype)


# ============================================================
# PYTORCH REFERENCE OUTPUT
# ============================================================

print("\n[3] Running PyTorch reference inference...")

with torch.no_grad():
    pytorch_output = model(input_tensor)

pytorch_np = pytorch_output.cpu().numpy().astype(np.float32)

print("[OK] PyTorch output shape:", pytorch_np.shape)
print(
    "[OK] PyTorch logits:",
    "min=", float(pytorch_np.min()),
    "max=", float(pytorch_np.max()),
    "mean=", float(pytorch_np.mean()),
    "std=", float(pytorch_np.std())
)


# ============================================================
# EXPORT REAL MODEL
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\n[4] Exporting real trained model:")
print(OUTPUT_PATH)

torch.onnx.export(
    model,
    input_tensor,
    str(OUTPUT_PATH),
    export_params=True,
    opset_version=OPSET_VERSION,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {
            0: "batch_size",
            2: "height",
            3: "width"
        },
        "output": {
            0: "batch_size",
            2: "height",
            3: "width"
        }
    }
)

print("[OK] ONNX export completed.")


# ============================================================
# VALIDATE ONNX GRAPH
# ============================================================

print("\n[5] Validating ONNX graph...")

onnx_model = onnx.load(str(OUTPUT_PATH))

onnx.checker.check_model(onnx_model)

print("[OK] ONNX checker passed.")
print("[OK] Node count:", len(onnx_model.graph.node))
print(
    "[OK] Ops:",
    sorted(set(node.op_type for node in onnx_model.graph.node))
)

if len(onnx_model.graph.node) <= 1:
    raise RuntimeError(
        "Exported ONNX contains only one or zero nodes. "
        "This is not the expected full segmentation model."
    )


# ============================================================
# ONNX RUNTIME INFERENCE
# ============================================================

print("\n[6] Running ONNX Runtime on EXACT SAME tensor...")

session = ort.InferenceSession(
    str(OUTPUT_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

onnx_outputs = session.run(
    None,
    {
        input_name: input_np
    }
)

onnx_np = np.asarray(
    onnx_outputs[0],
    dtype=np.float32
)

print("[OK] ONNX output shape:", onnx_np.shape)

print(
    "[OK] ONNX logits:",
    "min=", float(onnx_np.min()),
    "max=", float(onnx_np.max()),
    "mean=", float(onnx_np.mean()),
    "std=", float(onnx_np.std())
)


# ============================================================
# NUMERICAL EQUIVALENCE TEST
# ============================================================

print("\n[7] Comparing PyTorch vs ONNX...")

if pytorch_np.shape != onnx_np.shape:
    raise RuntimeError(
        f"Output shape mismatch: "
        f"PyTorch={pytorch_np.shape}, "
        f"ONNX={onnx_np.shape}"
    )

absolute_difference = np.abs(
    pytorch_np - onnx_np
)

max_difference = float(
    np.max(absolute_difference)
)

mean_difference = float(
    np.mean(absolute_difference)
)

print("Maximum absolute difference:", max_difference)
print("Mean absolute difference:", mean_difference)

# ONNX export should be numerically very close.
MAX_ALLOWED_DIFFERENCE = 1e-3

if max_difference > MAX_ALLOWED_DIFFERENCE:
    raise RuntimeError(
        "PyTorch and ONNX outputs are NOT numerically equivalent. "
        f"Maximum difference {max_difference} exceeds "
        f"allowed threshold {MAX_ALLOWED_DIFFERENCE}."
    )

print("[PASS] PyTorch and ONNX outputs are numerically equivalent.")


# ============================================================
# CLASS MAP EQUIVALENCE
# ============================================================

print("\n[8] Comparing segmentation class maps...")

pytorch_classes = np.argmax(
    pytorch_np,
    axis=1
).astype(np.int32)

onnx_classes = np.argmax(
    onnx_np,
    axis=1
).astype(np.int32)

class_map_difference = np.count_nonzero(
    pytorch_classes != onnx_classes
)

total_pixels = pytorch_classes.size

difference_percentage = (
    class_map_difference / total_pixels
) * 100.0

print("Different pixels:", class_map_difference)
print("Total pixels:", total_pixels)
print(
    "Class-map difference:",
    difference_percentage,
    "%"
)

if class_map_difference != 0:
    raise RuntimeError(
        "PyTorch and ONNX class maps differ."
    )

print("[PASS] PyTorch and ONNX class maps are identical.")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("EXPORT VALIDATION SUCCESSFUL")
print("=" * 70)

print("\nValidated ONNX:")
print(OUTPUT_PATH.resolve())

print("\nThe production ONNX has NOT been replaced.")
print("Only the temporary test ONNX was created.")