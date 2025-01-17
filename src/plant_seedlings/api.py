from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from enum import Enum
import cv2
from contextlib import asynccontextmanager
import torch
from .model import MyAwesomeModel, timm_model
import numpy as np
import json


@asynccontextmanager
async def lifespan(app: FastAPI):
    "Load and clean up model on startup and shutdown."
    try:
        print("Starting the app...Here we go!")

        print("Loading the models...")

        global custom_model, mobile_net, device, classes

        # Load the models and set weights
        custom_model = MyAwesomeModel()
        custom_model.load_state_dict(torch.load("models/custom/custom.pth"))
        mobile_net = timm_model("mobilenetv3_small_050", 12)
        mobile_net.load_state_dict(torch.load("models/mobilenetv3_small_050/mobilenetv3_small_050_30.pth"))
        resnet = timm_model("resnet18", 12)
        resnet.load_state_dict(torch.load("models/resnet18/resnet18_28.pth"))
        # TODO: add new model here

        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        custom_model.to(device)
        mobile_net.to(device)

        print("Models loaded successfully!")

        print("Loading the classes...")
        with open("models/classes.json", "r") as f:
            classes = json.load(f)

        print("Classes loaded successfully!")

        yield
        print("Cleaning up...")
        del custom_model, mobile_net, device, classes
        print("Shutting down the app...Goodbye!")
    finally:
        pass


class ModelEnum(Enum):
    # If you add any new model, be sure to add it to the api test as well.
    # Remember the test is order-sensitive
    custom = "custom"
    resnet = "resnet"
    mobilenet = "mobilenet"


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/models")
async def read_models():
    """Return the available models."""
    return {"models": list(ModelEnum)}


@app.post("/predict")
async def predict(model: ModelEnum, file: UploadFile = File(...)):
    """Predict the class of the image."""
    # Read the image from the uploaded file
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Resize the image to 224x224
    img = cv2.resize(img, (224, 224))

    # Convert the image to a tensor and normalize
    # now has shape (1, 3, 224, 224)
    # Normalization changes the range of pixel values from [0, 255] to [0, 1]
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img = img.to(device)

    # Select the appropriate model
    selected_model = custom_model if model == ModelEnum.custom else mobile_net

    # Make the prediction
    with torch.no_grad():
        selected_model.eval()
        output = selected_model(img)
        _, prediction = torch.max(output, 1)

    print(f"Prediction: {classes[str(prediction.item())]}")

    # Return the prediction
    return {"class": classes[str(prediction.item())]}
