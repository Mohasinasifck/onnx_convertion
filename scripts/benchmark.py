import os
import onnxruntime as ort
import numpy as np
import time

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../onnx_models/model.onnx"))
quantized_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../onnx_models/quantized_model.onnx"))

def benchmark_model(model_path, input_data, num_runs=100):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Warm-up run
    session.run(None, {input_name: input_data})

    # Measure latency over multiple runs
    total_time = 0
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: input_data})
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    avg_latency = (total_time / num_runs) * 1000  # Convert to ms
    print(f"Model: {model_path}, Average Latency: {avg_latency:.4f} ms over {num_runs} runs")
    return avg_latency

input_data = np.random.rand(1000, 3).astype(np.float32)

benchmark_model(model_path, input_data)
benchmark_model(quantized_model_path, input_data)
