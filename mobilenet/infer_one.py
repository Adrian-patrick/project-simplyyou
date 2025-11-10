import os
import sys
import json
import torch
import numpy as np
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Embedded label map (index mapping)
label_map = json.loads('{"biryani": 0, "burger": 1, "chai": 2, "chapati": 3, "cholebhature": 4, "dahl": 5, "dhokla": 6, "dosa": 7, "friedrice": 8, "gulabjamun": 9, "idli": 10, "jalebi": 11, "kaathirolls": 12, "kadaipaneer": 13, "kulfi": 14, "momos": 15, "naan": 16, "paanipuri": 17, "pakode": 18, "pavbhaji": 19, "pizza": 20, "poha": 21, "rolls": 22, "samosa": 23, "vadapav": 24}')
num_classes_loaded = len(label_map)

def load_model(model_path, num_classes, device):
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(weights=None)  # pretrained=False (new API uses weights=None)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def load_onnx(model_path):
    sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess

def preprocess(image_path):
    tfm = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    t = tfm(img).unsqueeze(0)  # [1,3,H,W]
    return t

def infer_torch(model, image_tensor, label_map):
    with torch.no_grad():
        logits = model(image_tensor)
    probs = torch.softmax(logits[0], dim=0)
    top5_prob, top5_idx = torch.topk(probs, 5)
    idx_to_class = {str(v): k for k, v in label_map.items()}
    print("Top 5 predictions (PyTorch):")
    for p, i in zip(top5_prob.tolist(), top5_idx.tolist()):
        print(f"  {idx_to_class.get(str(i), i)}: {p:.4f}")

def infer_onnx(sess, image_tensor, label_map):
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    ort_in = {inp_name: image_tensor.detach().cpu().numpy()}
    out = sess.run([out_name], ort_in)[0]  # [1,num_classes]
    import torch as _torch
    probs = _torch.softmax(_torch.tensor(out[0]), dim=0)
    top5_prob, top5_idx = _torch.topk(probs, 5)
    idx_to_class = {str(v): k for k, v in label_map.items()}
    print("Top 5 predictions (ONNX):")
    for p, i in zip(top5_prob.tolist(), top5_idx.tolist()):
        print(f"  {idx_to_class.get(str(i), i)}: {p:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python infer_one.py <model_path(.pth|.onnx)> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)

    # Preprocess
    try:
        image_tensor = preprocess(image_path)
        print(f"Preprocessed image tensor: {tuple(image_tensor.shape)} (expect (1,3,224,224))")
    except Exception as e:
        print(f"Preprocess failed: {e}")
        sys.exit(1)

    # Run
    if model_path.lower().endswith(".pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = load_model(model_path, num_classes_loaded, device)
            infer_torch(model, image_tensor.to(device), label_map)
        except Exception as e:
            print(f"PyTorch inference failed: {e}")
            sys.exit(1)

    elif model_path.lower().endswith(".onnx"):
        try:
            sess = load_onnx(model_path)
            infer_onnx(sess, image_tensor, label_map)
        except Exception as e:
            print(f"ONNX inference failed: {e}")
            sys.exit(1)
    else:
        print("Unsupported model extension. Use .pth or .onnx")
        sys.exit(1)
