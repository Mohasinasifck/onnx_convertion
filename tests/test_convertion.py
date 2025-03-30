import onnx

onnx_model = onnx.load("../onnx_models/model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
