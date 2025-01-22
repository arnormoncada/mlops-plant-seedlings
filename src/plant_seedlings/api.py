from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
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
from prometheus_client import Counter, make_asgi_app, CollectorRegistry, Histogram, Summary
# from hydra import compose, initialize

from .utils import extract_features

MY_REGISTRY = CollectorRegistry()
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Define prometheus metrics
error_counter = Counter("errors", "Number of errors", ["endpoint", "status_code"], registry=MY_REGISTRY)
request_counter = Counter("requests", "Number of requests", ["endpoint", "status_code"], registry=MY_REGISTRY)
request_latency = Histogram("request_latency_seconds", "Request latency", ["endpoint"], registry=MY_REGISTRY)
plant_image_size = Summary("plant_image_size", "Size of the plant image inputs", registry=MY_REGISTRY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    "Load and clean up model on startup and shutdown."
    try:
        print("Starting the app...Here we go!")

        print("Loading the models...")

        #TODO: Fix the hydra issue 
        # with initialize(config_path="../../configs", version_base="1.3"):
        #     cfg = compose(config_name="config.yaml")
    
        global custom_model, mobile_net, resnet, custom_model_path, mobile_net_path, resnet_path, device, classes

        # Load the models and set weights
        custom_model_path = "models/custom/custom_3.pth"
        mobile_net_path = "models/mobilenetv3_small_050/mobilenetv3_small_050_30.pth"
        resnet_path = "models/resnet18/resnet18_2.pth"

        custom_model = MyAwesomeModel()
        custom_model.load_state_dict(torch.load(custom_model_path))

        mobile_net = timm_model("mobilenetv3_small_050", 12)
        mobile_net.load_state_dict(torch.load(mobile_net_path))

        resnet = timm_model("resnet18", 12)
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
app.mount("/metrics", make_asgi_app(registry=MY_REGISTRY))

def add_to_database(
    features: np.array,
    target: int
) -> None:
    """Simple function to add prediction to database."""
    with open("data_drifting/prediction_database.csv", "a") as file:
        file.write(f"{features[0]}, {features[1]}, {features[2]}, {target}\n")


@app.get("/")
async def root():
    """Health check."""

    request_counter.labels(endpoint="root", status_code=HTTPStatus.OK).inc()

    with request_latency.labels("root").time():
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
        }
        return response


@app.get("/models")
async def read_models():
    """Return the available models."""

    request_counter.labels(endpoint="models", status_code=HTTPStatus.OK).inc()
    with request_latency.labels("models").time():
        return {"models": list(ModelEnum)}


@app.post(
        "/predict",
        response_model=dict, 
        responses={
            200: {"description": "Prediction successful", "content": {"application/json": {"example": {"class": "Loose Silky-bent", "model": "custom"}}}}, 
            400: {"description": "Bad request", "content": {"application/json": {"example": {"detail": "Allowed file types are png, jpg, jpeg"}}}},
            500: {"description": "Internal server error", "content": {"application/json": {"example": {"detail": "Internal server error"}}}}
        }
    )
async def predict(model: ModelEnum, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Predict the class of the image."""

    request_counter.labels(endpoint="predict", status_code=HTTPStatus.OK).inc()
    with request_latency.labels("predict").time():
        try:

            # Read the image from the uploaded file
            contents = await file.read()

            plant_image_size.observe(len(contents))

            # if file is not an image raise 400 error
            filename = file.filename
            if not filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                error_counter.labels(endpoint="predict", status_code=HTTPStatus.BAD_REQUEST).inc()
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Allowed file types are png, jpg, jpeg")


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
                # error_counter.labels(endpoint="predict", status_code=HTTPStatus.BAD_REQUEST).inc()
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Model not found")

            # Make the prediction
            with torch.no_grad():
                selected_model.eval()
                output = selected_model(img)
                _, prediction = torch.max(output, 1)

            # prediction = classes[str(prediction.item())]
            print(f"Prediction: {classes[str(prediction.item())]}")
            
            # Extract features
            img = img.numpy(force=True)
            features = extract_features(img)
            # print(features)
            background_tasks.add_task(add_to_database, features[0], int(prediction))

            # Return the prediction
            return {"class": classes[str(prediction.item())], "model": selected_model_path}
        except HTTPException as e:
            raise e
        except Exception as e:
            error_counter.labels(endpoint="predict", status_code=HTTPStatus.INTERNAL_SERVER_ERROR).inc()
            print(f"Error in prediction: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Internal server error")


# uvicorn --reload --port 8000 src.plant_seedlings.api:app
