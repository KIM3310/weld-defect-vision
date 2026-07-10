"""Tests for FastAPI detection endpoints."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastapi.testclient import TestClient
import pytest

from api.main import app
from edge.common.watchdog import check_http_health, validate_health_url

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestWatchdogHealthUrl:
    def test_validate_health_url_keeps_loopback_default(self):
        assert validate_health_url("http://localhost:8000/health") == "http://localhost:8000/health"

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
            healthy, detail = check_http_health(f"http://127.0.0.1:{server.server_port}/health", 1.0)

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
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/health")
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
