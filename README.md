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
Food Image Classification Model
This folder contains a fine-tuned MobileNetV2 model for Indian food classification.
⦁	PyTorch: best_mobilenetv2.pth
⦁	ONNX: mobilenetv2_224.onnx
⦁	Labels: label_map.json (index mapping used during training)
Test Metrics
⦁	Test Loss: 0.5900
⦁	Test Top-1 Accuracy: 83.39%
⦁	Test Top-3 Accuracy: 93.75%
Per-Class Accuracy
⦁	biryani: 77.27%
⦁	burger: 95.45%
⦁	chai: 100.00%
⦁	chapati: 82.61%
⦁	cholebhature: 91.30%
⦁	dahl: 95.45%
⦁	dhokla: 73.91%
⦁	dosa: 86.96%
⦁	friedrice: 72.73%
⦁	gulabjamun: 95.45%
⦁	idli: 91.30%
⦁	jalebi: 81.82%
⦁	kaathirolls: 59.09%
⦁	kadaipaneer: 59.09%
⦁	kulfi: 60.00%
⦁	momos: 82.61%
⦁	naan: 78.26%
⦁	paanipuri: 86.36%
⦁	pakode: 86.96%
⦁	pavbhaji: 86.36%
⦁	pizza: 82.61%
⦁	poha: 91.30%
⦁	rolls: 81.82%
⦁	samosa: 82.61%
⦁	vadapav: 100.00%


### 🚀 Deployment
- Self-contained Gradio app (`gradio_app.py`)
- deployed in [Hugging Face Spaces](https://huggingface.co/spaces)
