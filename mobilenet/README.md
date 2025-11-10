# Food Image Classification Model

This folder contains a fine-tuned MobileNetV2 model for Indian food classification.

## Models

* PyTorch: best\_mobilenetv2.pth
* ONNX: mobilenetv2\_224.onnx
* Labels: label\_map.json (index mapping used during training)

## Test Metrics

* Test Loss: 0.5900
* Test Top-1 Accuracy: 83.39%
* Test Top-3 Accuracy: 93.75%

### Per-Class Accuracy

* biryani: 77.27%
* burger: 95.45%
* chai: 100.00%
* chapati: 82.61%
* cholebhature: 91.30%
* dahl: 95.45%
* dhokla: 73.91%
* dosa: 86.96%
* friedrice: 72.73%
* gulabjamun: 95.45%
* idli: 91.30%
* jalebi: 81.82%
* kaathirolls: 59.09%
* kadaipaneer: 59.09%
* kulfi: 60.00%
* momos: 82.61%
* naan: 78.26%
* paanipuri: 86.36%
* pakode: 86.96%
* pavbhaji: 86.36%
* pizza: 82.61%
* poha: 91.30%
* rolls: 81.82%
* samosa: 82.61%
* vadapav: 100.00%

## Inference

### Requirements

* Python 3.9+
* pip install torch torchvision onnx onnxruntime pillow numpy

### Run

* PyTorch:
  python "C:\\Users\\adria\\OneDrive\\Desktop\\CODING\\simplyyou\\mobilenet\\infer\_one.py" "C:\\Users\\adria\\OneDrive\\Desktop\\CODING\\simplyyou\\mobilenet\\best\_mobilenetv2.pth" "C:\\path\\to\\image.jpg"
* ONNX:
  python "C:\\Users\\adria\\OneDrive\\Desktop\\CODING\\simplyyou\\mobilenet\\infer\_one.py" "C:\\Users\\adria\\OneDrive\\Desktop\\CODING\\simplyyou\\mobilenet\\mobilenetv2\_224.onnx" "C:\\path\\to\\image.jpg"

Notes:

* Input is resized to 224x224 with ImageNet normalization.
* For ONNX, CPUExecutionProvider is used by default; add CUDA provider if configured.
   

API INFERENCE

cd C:\\Users\\adria\\OneDrive\\Desktop\\CODING\\simplyyou

pip install fastapi uvicorn torch torchvision onnx onnxruntime pillow numpy

uvicorn app:app --host 0.0.0.0 --port 8000

CURL COMMAND FOR INFERENCE

CHAI

curl -X POST http://127.0.0.1:8000/infer -H "Content-Type: application/json" -d "{\\"image\_path\\":\\"C:\\\\\\\\Users\\\\\\\\adria\\\\\\\\OneDrive\\\\\\\\Desktop\\\\\\\\CODING\\\\\\\\simplyyou\\\\\\\\testimg\\\\\\\\testchai.jpeg\\",\\"runtime\\":\\"auto\\"}"

DOSA

curl -X POST http://127.0.0.1:8000/infer -H "Content-Type: application/json" -d "{\\"image\_path\\":\\"C:\\\\\\\\Users\\\\\\\\adria\\\\\\\\OneDrive\\\\\\\\Desktop\\\\\\\\CODING\\\\\\\\simplyyou\\\\\\\\testimg\\\\\\\\testdosa.jpeg\\",\\"runtime\\":\\"auto\\"}"




IDLI



curl -X POST http://127.0.0.1:8000/infer -H "Content-Type: application/json" -d "{\\"image\_path\\":\\"C:\\\\\\\\Users\\\\\\\\adria\\\\\\\\OneDrive\\\\\\\\Desktop\\\\\\\\CODING\\\\\\\\simplyyou\\\\\\\\testimg\\\\\\\\testidli.jpeg\\",\\"runtime\\":\\"auto\\"}"



PAKODE



curl -X POST http://127.0.0.1:8000/infer -H "Content-Type: application/json" -d "{\\"image\_path\\":\\"C:\\\\\\\\Users\\\\\\\\adria\\\\\\\\OneDrive\\\\\\\\Desktop\\\\\\\\CODING\\\\\\\\simplyyou\\\\\\\\testimg\\\\\\\\testpakode.jpeg\\",\\"runtime\\":\\"auto\\"}"


BIRYANI



curl -X POST http://127.0.0.1:8000/infer -H "Content-Type: application/json" -d "{\\"image\_path\\":\\"C:\\\\\\\\Users\\\\\\\\adria\\\\\\\\OneDrive\\\\\\\\Desktop\\\\\\\\CODING\\\\\\\\simplyyou\\\\\\\\testimg\\\\\\\\testbiryani.jpeg\\",\\"runtime\\":\\"auto\\"}"


