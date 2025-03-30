import os
import torch
import numpy as np
from models.model import SimpleModel  # Ensure this file exists
from scripts.convert_to_onnx import onnx_path
from scripts.benchmark import benchmark_model
from scripts.convert_to_onnx import onnx_path
print(f"ONNX Model Path: {onnx_path}")



# Define model and ONNX paths
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/trained_model.pth"))
ONNX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "onnx_models/model.onnx"))
QUANTIZED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "onnx_models/quantized_model.onnx"))

# Ensure model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Model file not found at {MODEL_PATH}. Please train the model first.")

# Load the trained model
model = SimpleModel()  # Initialize the model
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
print(f" Model loaded successfully from {MODEL_PATH}")

def train_and_save_model():
    """ Train a dummy model and save it """
    model = SimpleModel()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f" Trained model saved at {MODEL_PATH}")

def convert_model_to_onnx():
    """ Convert the trained model to ONNX format """
    os.system("python scripts/convert_to_onnx.py")
    if os.path.exists(ONNX_PATH):
        print(f" Model converted to ONNX and saved at {ONNX_PATH}")
    else:
        print(f" ONNX conversion failed! Check `scripts/convert_to_onnx.py`.")

def quantize_model():
    """ Apply quantization to ONNX model """
    os.system("python scripts/quantize_onnx.py")
    if os.path.exists(QUANTIZED_PATH):
        print(f" Quantized model saved at {QUANTIZED_PATH}")
    else:
        print(f" Quantization failed! Check `scripts/quantize_onnx.py`.")

def benchmark_models():
    """ Compare performance of ONNX and quantized ONNX model """
    input_data = np.random.rand(1, 3).astype(np.float32)
    print("\n Running Benchmark...")
    benchmark_model(ONNX_PATH, input_data)
    benchmark_model(QUANTIZED_PATH, input_data)

if __name__ == "__main__":
    train_and_save_model()
    convert_model_to_onnx()
    quantize_model()
    benchmark_models()
    print("\n Pipeline completed successfully!")
