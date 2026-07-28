"""Tests for FastAPI detection endpoints."""

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main
from api.main import INVALID_IMAGE_DETAIL, app
from edge.common.watchdog import check_http_health, validate_health_url

client = TestClient(app)


def image_bytes(width: int = 32, height: int = 32) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestWatchdogHealthUrl:
    def test_validate_health_url_keeps_loopback_default(self):
        assert (
            validate_health_url("http://localhost:8000/health")
            == "http://localhost:8000/health"
        )

    def test_validate_health_url_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="http or https"):
            validate_health_url("file:///etc/passwd")

    def test_validate_health_url_rejects_userinfo(self):
        with pytest.raises(ValueError, match="userinfo"):
            validate_health_url("https://user@example.com/health")

    def test_check_http_health_rejects_unsafe_redirect(self):
        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", "file:///etc/passwd")
                self.end_headers()

            def log_message(self, *_args):
                return None

        server = HTTPServer(("127.0.0.1", 0), RedirectHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            healthy, detail = check_http_health(
                f"http://127.0.0.1:{server.server_port}/health", 1.0
            )

            assert healthy is False
            assert "unsafe redirect" in detail
        finally:
            server.shutdown()
            server.server_close()

    def test_check_http_health_does_not_request_redirect_target(self):
        target_requests: list[str] = []

        class TargetHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                target_requests.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *_args):
                return None

        target = HTTPServer(("127.0.0.1", 0), TargetHandler)
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        target_thread.start()

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header(
                    "Location", f"http://127.0.0.1:{target.server_port}/health"
                )
                self.end_headers()

            def log_message(self, *_args):
                return None

        redirector = HTTPServer(("127.0.0.1", 0), RedirectHandler)
        redirect_thread = threading.Thread(target=redirector.serve_forever, daemon=True)
        redirect_thread.start()
        try:
            healthy, detail = check_http_health(
                f"http://127.0.0.1:{redirector.server_port}/health",
                1.0,
            )

            assert healthy is False
            assert "unsafe redirect" in detail
            assert target_requests == []
        finally:
            redirector.shutdown()
            redirector.server_close()
            target.shutdown()
            target.server_close()


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

    @pytest.mark.parametrize("path", ["/detect", "/detect/visualize"])
    def test_rejects_unidentified_image_before_model_loading(self, path):
        response = client.post(
            path,
            files={"file": ("broken.png", b"not really a png", "image/png")},
        )

        assert response.status_code == 422
        assert response.json() == {"detail": INVALID_IMAGE_DETAIL}

    @pytest.mark.parametrize("path", ["/detect", "/detect/visualize"])
    def test_rejects_upload_over_byte_limit(self, monkeypatch, path):
        monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 4)

        response = client.post(
            path,
            files={"file": ("large.png", b"12345", "image/png")},
        )

        assert response.status_code == 413
        assert response.json()["detail"] == "Image exceeds the 4-byte upload limit"

    @pytest.mark.parametrize("path", ["/detect", "/detect/visualize"])
    def test_rejects_upload_by_content_length_before_multipart_parsing(
        self, monkeypatch, path
    ):
        monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 4)
        monkeypatch.setattr(api_main, "MULTIPART_OVERHEAD_BYTES", 0)

        response = client.post(
            path,
            files={"file": ("small.png", b"x", "image/png")},
        )

        assert response.status_code == 413
        assert (
            response.json()["detail"] == "Request body exceeds the image upload limit"
        )

    @pytest.mark.parametrize("path", ["/detect", "/detect/visualize"])
    def test_rejects_streamed_upload_without_content_length(self, monkeypatch, path):
        monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 4)
        monkeypatch.setattr(api_main, "MULTIPART_OVERHEAD_BYTES", 0)
        boundary = "weld-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="weld.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        body += b"12345"
        body += f"\r\n--{boundary}--\r\n".encode()

        response = client.post(
            path,
            content=iter([body]),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )

        assert response.status_code == 413
        assert (
            response.json()["detail"] == "Request body exceeds the image upload limit"
        )

    @pytest.mark.parametrize("path", ["/detect", "/detect/visualize"])
    def test_rejects_excessive_decoded_pixels(self, monkeypatch, path):
        monkeypatch.setattr(api_main, "MAX_IMAGE_PIXELS", 4)

        response = client.post(
            path,
            files={"file": ("large-dimensions.png", image_bytes(), "image/png")},
        )

        assert response.status_code == 413
        assert response.json()["detail"] == "Image exceeds the 4-pixel limit"

    @pytest.mark.parametrize("path", ["/detect", "/detect/visualize"])
    def test_rejects_truncated_image_before_model_loading(self, path):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), color="white").save(buffer, format="PNG")
        truncated_png = buffer.getvalue()[:48]

        response = client.post(
            path,
            files={"file": ("truncated.png", truncated_png, "image/png")},
        )

        assert response.status_code == 422
        assert response.json() == {"detail": INVALID_IMAGE_DETAIL}

    def test_inference_concurrency_is_bounded(self, monkeypatch):
        class SlowDetector:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def detect(self, _image):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.05)
                with self.lock:
                    self.active -= 1
                return []

        detector = SlowDetector()
        monkeypatch.setattr(api_main, "get_detector", lambda: detector)
        monkeypatch.setattr(api_main, "_inference_slots", threading.BoundedSemaphore(1))

        def request_detection():
            with TestClient(app) as request_client:
                return request_client.post(
                    "/detect",
                    files={"file": ("weld.png", image_bytes(), "image/png")},
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(lambda _index: request_detection(), range(2)))

        assert [response.status_code for response in responses] == [200, 200]
        assert detector.max_active == 1
