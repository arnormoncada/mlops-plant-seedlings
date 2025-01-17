
from locust import HttpUser, between, task


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def get_models(self) -> None:
        """A task that simulates a user visiting the /models endpoint of the FastAPI app."""
        self.client.get("/models")

    @task(5)
    def predict_custom(self) -> None:
        """A task that simulates a user making a prediction using the custom model."""
        self.client.post(
            "/predict?model=custom", files={"file": ("test.jpg", open("tests/support/test_img.png", "rb"), "image/png")}
        )

    @task(5)
    def predict_mobilenet(self) -> None:
        """A task that simulates a user making a prediction using the mobilenet model."""
        self.client.post(
            "/predict?model=mobilenet",
            files={"file": ("test.jpg", open("tests/support/test_img.png", "rb"), "image/png")},
        )
