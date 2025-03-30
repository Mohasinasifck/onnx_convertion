import os
import sys
import torch

# Add project root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.model import SimpleModel  # Now it should work

# Define correct model path
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained_model.pth"))
print(f" Checking model path: {MODEL_PATH}")

# Verify if the file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f" Model file NOT FOUND at {MODEL_PATH}. Run `python main.py` to train first.")

# Load model
model = SimpleModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
print(" Model loaded successfully!")

# Save as ONNX
ONNX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../onnx_models/model.onnx"))
dummy_input = torch.randn(1, 3)
dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
torch.onnx.export(model, dummy_input, "onnx_models/model.onnx", export_params=True,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes=dynamic_axes)

print(f" Model converted to ONNX: {ONNX_PATH}")
onnx_path = "C:/Users/mohas/ai_model_convertion/onnx_models/model.onnx"

