import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms
from fastapi.responses import HTMLResponse
import io
import os
import sys

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model_cnn import MyCNN

model = MyCNN()
model.load_state_dict(torch.load("../model/best_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h2>FastAPI Cat & Dog Classification API is running!</h2>"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Baca file upload
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Transform & kirim ke device
        img = transform(img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(img)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        return {
            "prediction": int(pred),
            "confidence": float(probs[0][pred])
        }

    except Exception as e:
        return {"error": str(e)}