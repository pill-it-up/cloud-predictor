from PIL import Image
from torchvision import models, transforms
from pathlib import Path

import torch
import json

import logging

logging.basicConfig(
    filename="{}".format(Path.home() / "logs" / "predictor.log"),
    format="%(asctime)s == PILLITUP == PRED_LOGIC == [%(levelname)-8s] %(message)s",
    level=logging.DEBUG,
)

MODEL = "model_black"

config = json.load((Path(__file__).resolve().parent / "models.json").open())

filepath = Path(__file__).resolve().parent / config[MODEL]["path"]
CLASSES = config[MODEL]["classes"]
N_CLASSES = len(CLASSES)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose(
    [transforms.Scale(224), transforms.ToTensor(), normalize]
)

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
    fc_out = model_ft(img_tensor)
    _, indices = torch.max(fc_out, 1)

    return CLASSES[indices[0]]
