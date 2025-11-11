---
title: Simplyou
emoji: 🍛
sdk: gradio
app_file: gradio_app.py
python_version: 3.11
---


# 🍛 Simplyou – Indian Food Classifier 🇮🇳

A fine-tuned EfficientNet / MobileNet model that classifies 25 popular Indian dishes  
and displays their nutrition information (calories, protein, carbs, and fats).

### ⚙️ Features
- Upload an Indian food image  
- Detect top-3 predicted dishes  
- Display nutritional macros per 100g  
- Built using `Gradio`, `ONNXRuntime`, and `PyTorch`

### 🧠 Model
The model is exported to ONNX format and loaded directly in the Gradio app.  
Class labels are stored in `mobilenet/label_map.json`.

### 🚀 Deployment
- Self-contained Gradio app (`gradio_app.py`)
- Ready to deploy to [Hugging Face Spaces](https://huggingface.co/spaces)
