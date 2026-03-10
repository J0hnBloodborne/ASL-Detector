import io
import json
import base64

import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from pydantic import BaseModel

app = FastAPI(title="ASL Object Detector")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Globals
model = None
label_map = None
device = None
CONFIDENCE_THRESHOLD = 0.5


def load_model():
    global model, label_map, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label map
    with open("train/label_map.json", "r") as f:
        label_map = json.load(f)

    # Build model
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 27)

    # Load weights
    model.load_state_dict(torch.load("train/model.pth", map_location=device))
    model.to(device)
    model.eval()
    import sys
    print(f"Model loaded on {device}", flush=True)
    sys.stdout.flush()


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class ImagePayload(BaseModel):
    image: str  # base64 encoded image


@app.post("/predict")
async def predict(payload: ImagePayload):
    global model, label_map, device

    img_data = payload.image
    if "," in img_data:
        img_data = img_data.split(",", 1)[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Resize to 640x640 (matches training resolution) and convert
    img = img.resize((640, 640))
    img_tensor = T.ToTensor()(img).to(device)

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Filter by confidence threshold
    results = []
    scores = predictions["scores"].cpu()
    labels = predictions["labels"].cpu()
    boxes = predictions["boxes"].cpu()

    for score, label, box in zip(scores, labels, boxes):
        if score.item() >= CONFIDENCE_THRESHOLD:
            label_idx = str(label.item())
            letter = label_map.get(label_idx, "?")
            results.append({
                "label": letter,
                "confidence": round(score.item(), 3),
                "box": [round(b, 1) for b in box.tolist()]
            })

    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
