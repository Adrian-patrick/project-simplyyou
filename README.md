# 🍛 Simplyou – Dockerized Indian Food Classifier API 🇮🇳

A **FastAPI + ONNXRuntime** based API that classifies Indian food images and provides nutritional macros.  
Fully **Dockerized** for easy deployment.

---

## ⚙️ Overview
- 🧠 Model: Fine-tuned **MobileNetV2 (ONNX)**
- 🚀 Backend: **FastAPI**
- 🧾 Output: Top-3 food predictions with confidence
- 🐳 Deployable via **Docker**, **Render**, or **Railway**

---

## 📂 Project Structure
```
📦 simplyyou/
 ┣ 📂 mobilenet/      → model + label_map.json
 ┣ app.py             → FastAPI app
 ┣ Dockerfile         → container instructions
 ┣ requirements.txt   → dependencies
 ┗ class_macros.csv   → nutritional info
```

---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Access Swagger UI → http://127.0.0.1:8000/docs

---

## 🐳 Run with Docker
```bash
docker build -t simplyyou-api .
docker run --rm -p 8000:8000 ^
  -e ART_DIR=/app/mobilenet ^
  -v "C:\Users\adria\Desktop\CODING\simplyyou\mobilenet":/app/mobilenet:ro ^
  simplyyou-api
```

---

## 🧠 Example API Response
```json
{
  "ok": true,
  "runtime": "onnx",
  "topk": [
    {"class": "biryani", "confidence": 0.92},
    {"class": "poha", "confidence": 0.04},
    {"class": "pavbhaji", "confidence": 0.02}
  ]
}
```

---

## 🌐 Deploy on Render
1. Push this branch (`dockerized`) to GitHub  
2. Create a **new Web Service** on [Render.com](https://render.com)  
3. Choose **Environment: Docker**, Port → `8000`  
4. Deploy 🚀  

---

**Author:** [Adrian Patrick](https://github.com/Adrian-patrick)  
🧠 *ML Engineer | AI Developer*  
