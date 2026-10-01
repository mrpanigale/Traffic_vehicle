"""Fast-API - Image Prediction API"""

# =============import=============
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image

from predict import predict_image, predict_image_hf

# ==========app===========
app = FastAPI(
    title="Vehicles Classification API",
    description="image classification with trained model",
    version="1.0.0",
)


# ===========prediction========
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model: Literal["resnet18", "hugging_face"] = Form("resnet18"),
):
    """Predict the class of an uploaded image."""

    try:
        img = Image.open(image.file).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )
    if model == "resnet18":
        result = predict_image(img)
    elif model == "hugging_face":
        result = predict_image_hf(img)

    return result
