import gradio as gr
import onnxruntime
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json
import pandas as pd

# ------------------------------------------------------------------
# Load label map and macros
# ------------------------------------------------------------------
with open("mobilenet/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
idx_to_class = {v: k for k, v in label_map.items()}

macros_df = pd.read_csv("class_macros.csv")
macros_df.columns = macros_df.columns.str.strip().str.lower()
food_macros = {
    row["class"].strip().lower(): {
        "calories": row["kcal_100g"],
        "protein": row["protein_g_100g"],
        "carbs": row["carbs_g_100g"],
        "fats": row["fat_g_100g"],
    }
    for _, row in macros_df.iterrows()
}

# ------------------------------------------------------------------
# Load ONNX model
# ------------------------------------------------------------------
MODEL_PATH = "mobilenet/mobilenetv2_224.onnx"
session = onnxruntime.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# ------------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------------
IMG_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess(image: Image.Image):
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor.numpy()

# ------------------------------------------------------------------
# Inference function
# ------------------------------------------------------------------
def predict(image):
    if image is None:
        return [], "Please upload an image."

    inp = preprocess(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    preds = session.run([output_name], {input_name: inp})[0]
    probs = torch.softmax(torch.tensor(preds[0]), dim=0)

    topk = torch.topk(probs, 3)
    top3_indices = topk.indices.tolist()
    top3_probs = topk.values.tolist()

    results = []
    for idx, prob in zip(top3_indices, top3_probs):
        cls_name = idx_to_class.get(idx, str(idx))
        results.append((cls_name, float(prob)))

    choices = [f"{cls_name} ({prob*100:.1f}%)" for cls_name, prob in results]
    return results, choices

# ------------------------------------------------------------------
# Display macros
# ------------------------------------------------------------------
def show_macros(selected_food):
    if not selected_food:
        return "Please select a food item."

    key = selected_food.strip().lower()
    if key not in food_macros:
        return f"No macros found for {selected_food}."

    macros = food_macros[key]
    result = f"""
    🍽️ **{selected_food.title()}**

    🔥 **Calories:** {macros['calories']} kcal  
    💪 **Protein:** {macros['protein']} g  
    🥔 **Carbs:** {macros['carbs']} g  
    🧈 **Fats:** {macros['fats']} g
    """
    return result

# ------------------------------------------------------------------
# Gradio Interface
# ------------------------------------------------------------------
with gr.Blocks(title="Simplyou - Indian Food Classifier") as demo:
    gr.Markdown("# 🍛 Simplyou - Indian Food Classifier 🇮🇳")
    gr.Markdown("Upload an Indian food image to detect what dish it is and view its nutrition info (macros).")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Food Image")

    predict_button = gr.Button("🍽️ Classify Food")
    prediction_choices = gr.Radio(label="Top 3 Predictions", choices=[])
    macros_output = gr.Markdown()

    def on_predict(image):
        results, choices = predict(image)
        if not results:
            return gr.update(choices=[], value=None), "Error: No predictions."
        labels = [r[0] for r in results]
        return gr.update(choices=labels, value=None), "Select the correct food from predictions above."

    predict_button.click(fn=on_predict, inputs=image_input, outputs=[prediction_choices, macros_output])
    prediction_choices.change(fn=show_macros, inputs=prediction_choices, outputs=macros_output)

if __name__ == "__main__":
    demo.launch()
