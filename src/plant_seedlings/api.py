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
from hydra import compose, initialize


@asynccontextmanager
async def lifespan(app: FastAPI):
    "Load and clean up model on startup and shutdown."
    try:
        print("Starting the app...Here we go!")

        print("Loading the models...")

        with initialize(config_path="../../configs", version_base="1.3"):
            cfg = compose(config_name="config.yaml")
    
        global custom_model, mobile_net, resnet, custom_model_path, mobile_net_path, resnet_path, device, classes

        # Load the models and set weights
        custom_model_path = "models/custom/custom_3.pth"
        mobile_net_path = "models/mobilenetv3_small_050/mobilenetv3_small_050_30.pth"
        resnet_path = "models/resnet18/resnet18_2.pth"

        custom_model = MyAwesomeModel()
        custom_model.load_state_dict(torch.load(custom_model_path))

        mobile_net = timm_model("mobilenetv3_small_050", cfg.num_classes)
        mobile_net.load_state_dict(torch.load(mobile_net_path))

        resnet = timm_model("resnet18", cfg.num_classes)
        resnet.load_state_dict(torch.load(resnet_path))
        

        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        custom_model.to(device)
        mobile_net.to(device)
        resnet.to(device)

        print("Models loaded successfully!")

        print("Loading the classes...")
        with open("models/classes.json", "r") as f:
            classes = json.load(f)

        print("Classes loaded successfully!")

        yield
        print("Cleaning up...")
        del custom_model, mobile_net, resnet, custom_model_path, mobile_net_path, resnet_path, device, classes
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
    if model == ModelEnum.custom:
        selected_model = custom_model
        selected_model_path = custom_model_path
    elif model == ModelEnum.mobilenet:
        selected_model = mobile_net
        selected_model_path = mobile_net_path
    elif model == ModelEnum.resnet:
        selected_model = resnet
        selected_model_path = resnet_path
    else:
        return {"error": "Invalid model"}

    # Make the prediction
    with torch.no_grad():
        selected_model.eval()
        output = selected_model(img)
        _, prediction = torch.max(output, 1)

    print(f"Prediction: {classes[str(prediction.item())]}")

    # Return the prediction
    return {"class": classes[str(prediction.item())], "model": selected_model_path}
