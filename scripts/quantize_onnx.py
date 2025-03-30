import os
import onnxruntime.quantization as quant

onnx_model_path = os.path.abspath("onnx_models/model.onnx")
quantized_model_path = os.path.abspath("onnx_models/quantized_model.onnx")

print(f" Loading ONNX model from: {onnx_model_path}")

quant.quantize_dynamic(onnx_model_path, quantized_model_path)

print(f"Quantized model saved at: {quantized_model_path}")
