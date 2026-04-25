"""Tests for FastAPI detection endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestClassesEndpoint:
    def test_get_classes(self):
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert len(data["classes"]) == 5


class TestDetectEndpoint:
    def test_detect_no_file(self):
        response = client.post("/detect")
        assert response.status_code == 422

    def test_detect_invalid_file_type(self):
        response = client.post(
            "/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_detect_rejects_corrupt_image_payload(self):
        response = client.post(
            "/detect",
            files={"file": ("broken.png", b"not really a png", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "File must be a valid image"

    def test_visualize_rejects_corrupt_image_payload(self):
        response = client.post(
            "/detect/visualize",
            files={"file": ("broken.png", b"not really a png", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "File must be a valid image"
