from fastapi.testclient import TestClient
from src.plant_seedlings.api import app
from typing import Dict

# Define expected response structure
predict_response_structure: Dict[str, type] = {
    "class": str,
}

# client = TestClient(app)


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "OK", "status-code": 200}


def test_read_models():
    with TestClient(app) as client:
        response = client.get("/models")
        assert response.status_code == 200
        assert response.json() == {"models": ["custom", "resnet", "mobilenet"]}


def validate_predict_response(response: Dict[str, str]) -> None:
    """helper function to validate the predict"""
    assert isinstance(response, dict)  # Check if response is a dictionary
    assert set(response.keys()) == set(predict_response_structure.keys())  # Check if response has the correct keys
    assert isinstance(response["class"], str)  # Check if the class is a string


def test_predict_custom():
    with TestClient(app) as client:
        response = client.post(
            "/predict?model=custom", files={"file": ("test.jpg", open("tests/support/test_img.png", "rb"), "image/png")}
        )
        assert response.status_code == 200
        validate_predict_response(response.json())


def test_predict_mobilenet():
    with TestClient(app) as client:
        response = client.post(
            "/predict?model=mobilenet",
            files={"file": ("test.jpg", open("tests/support/test_img.png", "rb"), "image/png")},
        )
        assert response.status_code == 200
        validate_predict_response(response.json())


def test_predict_resnet():
    with TestClient(app) as client:
        response = client.post(
            "/predict?model=resnet",
            files={"file": ("test.jpg", open("tests/support/test_img.png", "rb"), "image/png")},
        )
        assert response.status_code == 200
        validate_predict_response(response.json())
