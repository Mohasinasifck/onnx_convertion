
# **AI Model Conversion & Deployment**  

🚀 This project focuses on converting a trained AI model from **PyTorch** to **ONNX**, optimizing it through **quantization**, and benchmarking its performance. The final model is deployed using **CI/CD pipelines** with Docker & GitHub Actions.

---

## **📌 Features**  
✅ Convert PyTorch models to ONNX  
✅ Apply ONNX model quantization for optimization  
✅ Benchmark model performance  
✅ Automate deployment using GitHub Actions  
✅ Docker containerization for easy deployment  

---

## **📂 Project Structure**  
```
📦 ai_model_convertion
├── 📁 models             # Trained model files
├── 📁 onnx_models        # ONNX and quantized models
├── 📁 scripts            # Conversion, quantization, benchmarking scripts
│   ├── convert_to_onnx.py
│   ├── quantize_onnx.py
│   ├── benchmark.py
├── 📁 .github
│   ├── 📁 workflows      # CI/CD pipeline configurations
│       ├── ci_cd_pipeline.yml
├── main.py               # Entry point for conversion & benchmarking
├── Dockerfile            # Docker configuration for deployment
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## **🛠 Setup & Installation**

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/Mohasinasifck/onnx_convertion.git
cd your-repo
```

### **2️⃣ Create a Virtual Environment**  
```sh
conda create --name ai_deployment python=3.8 -y
conda activate ai_deployment
```

### **3️⃣ Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **4️⃣ Convert PyTorch Model to ONNX**  
```sh
python scripts/convert_to_onnx.py
```

### **5️⃣ Quantize the ONNX Model**  
```sh
python scripts/quantize_onnx.py
```

### **6️⃣ Benchmark Model Performance**  
```sh
python scripts/benchmark.py
```

---

## **🚀 Running with Docker**

### **1️⃣ Build the Docker Image**  
```sh
docker build -t ai-model .
```

### **2️⃣ Run the Container**  
```sh
docker run --rm -p 5000:5000 ai-model
```

---

## **⚡ CI/CD Pipeline**  
This project includes a **GitHub Actions CI/CD pipeline** to automate model conversion, quantization, and deployment. Ensure your workflow file is inside `.github/workflows/`.

### **🔹 To Trigger the Pipeline:**
1. **Push code to GitHub:**
   ```sh
   git add .
   git commit -m "Updated pipeline"
   git push origin main
   ```
2. **Check the GitHub Actions Tab** to monitor the pipeline execution.

---


🚀 **Happy Coding!** 🎯

