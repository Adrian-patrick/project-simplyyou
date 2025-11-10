import os
import sys
import json
import torch
import onnxruntime
from PIL import Image
import torchvision.transforms as T
import tempfile

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
import uvicorn

# ------------------------------------------------------------------
# Artifact paths (container-friendly)
# ------------------------------------------------------------------
ART_DIR = os.getenv("ART_DIR", "/app/mobilenet")

LABEL_MAP_JSON   = os.path.join(ART_DIR, "label_map.json")
ONNX_WEIGHTS     = os.path.join(ART_DIR, "mobilenetv2_224.onnx")
PYTORCH_WEIGHTS  = os.path.join(ART_DIR, "best_mobilenetv2.pth")

IMG_HEIGHT = 224
IMG_WIDTH  = 224
TOP_K      = 5

# Globals initialized at startup
class_to_idx = {}
idx_to_class = {}
num_classes = 0

# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
preprocess = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ------------------------------------------------------------------
# Model loaders (lazy)
# ------------------------------------------------------------------
_torch_model = None
def _load_torch_model(weights_path: str, num_classes: int):
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(weights=None)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_f, num_classes)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

_onnx_session = None
def _load_onnx_session(weights_path: str):
    sess = onnxruntime.InferenceSession(
        weights_path,
        providers=["CPUExecutionProvider"]
    )
    return sess

def _ensure_model_ready(prefer: str = "onnx"):
    global _torch_model, _onnx_session
    if prefer == "onnx" and os.path.exists(ONNX_WEIGHTS):
        if _onnx_session is None:
            _onnx_session = _load_onnx_session(ONNX_WEIGHTS)
        return "onnx"
    if prefer == "torch" and os.path.exists(PYTORCH_WEIGHTS):
        if _torch_model is None:
            _torch_model = _load_torch_model(PYTORCH_WEIGHTS, num_classes)
        return "torch"
    if os.path.exists(ONNX_WEIGHTS):
        if _onnx_session is None:
            _onnx_session = _load_onnx_session(ONNX_WEIGHTS)
        return "onnx"
    if os.path.exists(PYTORCH_WEIGHTS):
        if _torch_model is None:
            _torch_model = _load_torch_model(PYTORCH_WEIGHTS, num_classes)
        return "torch"
    raise RuntimeError("No model file found at expected paths inside ART_DIR.")

# ------------------------------------------------------------------
# Inference utilities
# ------------------------------------------------------------------
def _load_image_tensor(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    t = preprocess(img).unsqueeze(0)
    if t.shape != (1, 3, IMG_HEIGHT, IMG_WIDTH):
        raise ValueError(f"Tensor shape mismatch: {tuple(t.shape)}")
    return t

def _infer_torch(image_tensor: torch.Tensor):
    global _torch_model
    if _torch_model is None:
        _torch_model = _load_torch_model(PYTORCH_WEIGHTS, num_classes)
    with torch.no_grad():
        logits = _torch_model(image_tensor)
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise RuntimeError(f"Logits shape mismatch: {tuple(logits.shape)}")
    probs = torch.softmax(logits[0], dim=0)
    top_prob, top_idx = torch.topk(probs, k=min(TOP_K, num_classes))
    results = []
    for p, i in zip(top_prob.tolist(), top_idx.tolist()):
        results.append({"class": idx_to_class.get(int(i), str(int(i))), "index": int(i), "confidence": float(p)})
    return results

def _infer_onnx(image_tensor: torch.Tensor):
    global _onnx_session
    if _onnx_session is None:
        _onnx_session = _load_onnx_session(ONNX_WEIGHTS)
    inp_name = _onnx_session.get_inputs()[0].name
    out_name = _onnx_session.get_outputs()[0].name
    inp = {inp_name: image_tensor.detach().cpu().numpy()}
    out = _onnx_session.run([out_name], inp)[0]
    if out.ndim != 2 or out.shape[1] != num_classes:
        raise RuntimeError(f"ONNX output shape mismatch: {out.shape}")
    probs = torch.softmax(torch.tensor(out[0]), dim=0)
    top_prob, top_idx = torch.topk(probs, k=min(TOP_K, num_classes))
    results = []
    for p, i in zip(top_prob.tolist(), top_idx.tolist()):
        results.append({"class": idx_to_class.get(int(i), str(int(i))), "index": int(i), "confidence": float(p)})
    return results

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="Food Image Inference API", version="1.0.0")

class InferRequest(BaseModel):
    image_path: str = Field(..., description="Absolute or relative filesystem path to an image")
    runtime: str = Field("auto", description="'auto' | 'onnx' | 'torch'")

class InferResponse(BaseModel):
    ok: bool
    runtime: str
    image_path: str
    topk: list

class InferUploadResponse(BaseModel):
    ok: bool
    runtime: str
    topk: list

@app.get("/health")
def health():
    return {"ok": True, "classes": num_classes, "art_dir": ART_DIR}

@app.on_event("startup")
def _startup():
    global class_to_idx, idx_to_class, num_classes
    if not os.path.exists(LABEL_MAP_JSON):
        raise RuntimeError(f"label_map.json not found at {LABEL_MAP_JSON}")
    with open(LABEL_MAP_JSON, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    if not isinstance(class_to_idx, dict):
        raise RuntimeError("label_map.json must be a dict mapping class_name -> index.")
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
    if num_classes < 2:
        raise RuntimeError(f"label_map contains fewer than 2 classes: {num_classes}")

    runtime = _ensure_model_ready()
    print(f"[startup] runtime={runtime}, classes={num_classes}")
    print(f"[paths] onnx={'OK' if os.path.exists(ONNX_WEIGHTS) else 'MISSING'} :: {ONNX_WEIGHTS}")
    print(f"[paths] pth={'OK' if os.path.exists(PYTORCH_WEIGHTS) else 'MISSING'} :: {PYTORCH_WEIGHTS}")
    print(f"[paths] labels={'OK' if os.path.exists(LABEL_MAP_JSON) else 'MISSING'} :: {LABEL_MAP_JSON}")

@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    try:
        runtime = req.runtime.lower().strip()
        if runtime not in {"auto", "onnx", "torch"}:
            raise HTTPException(status_code=400, detail="runtime must be 'auto' | 'onnx' | 'torch'")

        if runtime == "onnx":
            if not os.path.exists(ONNX_WEIGHTS):
                raise HTTPException(status_code=400, detail=f"ONNX model not found at {ONNX_WEIGHTS}")
            _ensure_model_ready("onnx")
            chosen = "onnx"
        elif runtime == "torch":
            if not os.path.exists(PYTORCH_WEIGHTS):
                raise HTTPException(status_code=400, detail=f"PyTorch model not found at {PYTORCH_WEIGHTS}")
            _ensure_model_ready("torch")
            chosen = "torch"
        else:
            chosen = _ensure_model_ready()

        image_tensor = _load_image_tensor(req.image_path)

        if chosen == "onnx":
            preds = _infer_onnx(image_tensor)
        else:
            preds = _infer_torch(image_tensor)

        return InferResponse(ok=True, runtime=chosen, image_path=req.image_path, topk=preds)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")

@app.post("/infer_upload", response_model=InferUploadResponse)
async def infer_upload(file: UploadFile = File(...), runtime: str = "auto"):
    tmp_path = None
    try:
        runtime = runtime.lower().strip()
        if runtime not in {"auto", "onnx", "torch"}:
            raise HTTPException(status_code=400, detail="runtime must be 'auto' | 'onnx' | 'torch'")

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Run inference
        image_tensor = _load_image_tensor(tmp_path)
        
        if runtime == "onnx":
            _ensure_model_ready("onnx")
            chosen = "onnx"
        elif runtime == "torch":
            _ensure_model_ready("torch")
            chosen = "torch"
        else:
            chosen = _ensure_model_ready()

        if chosen == "onnx":
            preds = _infer_onnx(image_tensor)
        else:
            preds = _infer_torch(image_tensor)

        return InferUploadResponse(ok=True, runtime=chosen, topk=preds)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ------------------------------------------------------------------
# CLI mode (optional): python app.py <torch|onnx> <image_path>
# ------------------------------------------------------------------
def _cli():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  FastAPI server: python app.py")
        print("  CLI inference:  python app.py <torch|onnx> <image_path>")
        sys.exit(1)

    runtime = sys.argv[1].lower()
    image_path = sys.argv[2]

    if runtime not in {"torch", "onnx"}:
        print("runtime must be 'torch' or 'onnx'")
        sys.exit(1)

    image_tensor = _load_image_tensor(image_path)

    if runtime == "onnx":
        if not os.path.exists(ONNX_WEIGHTS):
            print(f"ONNX model not found at {ONNX_WEIGHTS}")
            sys.exit(1)
        _ensure_model_ready("onnx")
        results = _infer_onnx(image_tensor)
    else:
        if not os.path.exists(PYTORCH_WEIGHTS):
            print(f"PyTorch weights not found at {PYTORCH_WEIGHTS}")
            sys.exit(1)
        _ensure_model_ready("torch")
        results = _infer_torch(image_tensor)

    print("Top predictions:")
    for r in results:
        print(f"{r['class']}: {r['confidence']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
    else:
        _cli()
