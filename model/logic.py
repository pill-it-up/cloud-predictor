from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F
from pathlib import Path
from toolz import merge

import torch
import json
import numpy as np

import logging

MODEL = "model_black"

config = json.load((Path(__file__).resolve().parent / "models.json").open())

filepath = Path(__file__).resolve().parent / config[MODEL]["path"]
CLASSES = config[MODEL]["classes"]
N_CLASSES = len(CLASSES)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.Scale(224), transforms.ToTensor(), normalize])

upsample = torch.nn.Upsample((224, 224))


def model_load():
    logging.debug("Loading model.")
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, N_CLASSES)

    saved_state = torch.load(filepath, map_location=lambda storage, loc: storage)
    model_ft.load_state_dict(saved_state)
    model_ft.eval()
    logging.debug("Model loaded.")
    return model_ft


def preprocess_pil(img_pil):
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    img_tensor = upsample(img_tensor)
    return img_tensor


def predict(model_ft, img_file):
    logging.debug("Starting prediction.")
    img_pil = Image.open(img_file)
    logging.debug("Preprocessing image.")
    img_tensor = preprocess_pil(img_pil)
    logging.debug("Predicting.")
    fc_out = model_ft(img_tensor)

    with torch.no_grad():
        probs = F.softmax(fc_out)[0].numpy()

    def dictify_one(label, prob):
        return {label: str(prob)}

    all_probs_dict = merge(map(dictify_one, CLASSES, probs))
    _, indices = torch.max(fc_out, 1)

    return CLASSES[indices[0]], all_probs_dict


if __name__ == "__main__":
    model_ft = model_load()
    classes = predict(model_ft, "sample_imgs/pill_sample.jpg")
