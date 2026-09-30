"""Predict.py - Predict with best model on test data"""

#==========imports===========
from torchvision.models import resnet18
import torch
from sys import exit
import torch.nn as nn
from data_provider import ROOT,get_transform
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_path = ROOT / "models" / "resnet_finetune" / "resnet_finetune.pth"
_,transform = get_transform()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['ambulance', 'autobus', 'kamyun', 'kamyunet', 'minibus', 'savari', 'taxi', 'vanet']
#=========functions============
def predict_image(image,transform_base=transform):
    """Predict with best model on test data"""
    image = transform_base(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
    proba = torch.softmax(output, dim=1)[0]
    predict_idx = proba.argmax().item()

    result = {
        "predicted_class": classes[predict_idx],
        "confidence": float(proba[predict_idx]),
        "probabilities": {
            class_name: float(prob)
            for class_name, prob in zip(classes, proba)
        },
        "needs_review": float(proba[predict_idx]) < 0.7
    }
    return result

def predict_image_hf(image):
    """Predict an image with the fine-tuned Hugging Face model."""

    inputs = hf_processor(
        image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = hf_model(**inputs)

    proba = torch.softmax(output.logits, dim=1)[0]
    predict_idx = proba.argmax().item()

    result = {
        "predicted_class": classes[predict_idx],
        "confidence": float(proba[predict_idx]),
        "probabilities": {
            class_name: float(prob)
            for class_name, prob in zip(classes, proba)
        },
        "needs_review": float(proba[predict_idx]) < 0.7
    }

    return result
#==========load-model===========
model = resnet18(weights = None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(classes))

try:
    state = torch.load(model_path, weights_only=True)
except FileNotFoundError:
    print("Did you run all models first?")
    exit(1)

model.load_state_dict(state)
model.to(device)
model.eval()

print("<model loaded>")

#=========hugging-face-model=========

hf_model_path = ROOT / "models" / "hugging_face_model" / "best_model"

hf_processor = AutoImageProcessor.from_pretrained(hf_model_path)
hf_model = AutoModelForImageClassification.from_pretrained(hf_model_path)

hf_model.to(device)
hf_model.eval()


