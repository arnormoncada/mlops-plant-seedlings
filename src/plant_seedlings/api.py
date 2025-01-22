from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from enum import Enum
import cv2
from contextlib import asynccontextmanager
import torch
from .model import MyAwesomeModel, timm_model
import numpy as np
import json
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


@asynccontextmanager
async def lifespan(app: FastAPI):
    "Load and clean up model on startup and shutdown."
    try:
        print("Starting the app...Here we go!")

        print("Loading the models...")

        global custom_model, mobile_net, device, classes, resnet

        # Load the models and set weights
        custom_model = MyAwesomeModel()
        custom_model.load_state_dict(torch.load("models/custom/custom_3.pth"))
        mobile_net = timm_model("mobilenetv3_small_050", 12)
        mobile_net.load_state_dict(torch.load("models/mobilenetv3_small_050/mobilenetv3_small_050_1.pth"))
        resnet = timm_model("resnet18", 12)
        resnet.load_state_dict(torch.load("models/resnet18/resnet18_2.pth"))

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

    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img)

   # Preprocess the image
    transform = Compose([Resize((224, 224)), ToTensor()])

    img = transform(pil_img).unsqueeze(0).to(device)

    # Select the appropriate model
    selected_model = (
        custom_model if model == ModelEnum.custom 
        else mobile_net 
        if model == ModelEnum.mobilenet 
        else resnet
    )
    # print chosen model name
    print(f"Model chosen: {selected_model}")

    # Make the prediction
    with torch.no_grad():
        selected_model.eval()
        output = selected_model(img)
        _, prediction = torch.max(output, 1)

    print(f"Prediction: {classes[str(prediction.item())]}")

    # Return the prediction
    return {"class": classes[str(prediction.item())]}
