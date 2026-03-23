"""Tests for the FastAPI inspection API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from tests.conftest import (
    image_to_bytes,
    make_image_crack,
    make_image_no_defect,
    make_image_porosity,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


class TestRootEndpoint:
    def test_root_returns_200(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_service_key(self, client: TestClient) -> None:
        response = client.get("/")
        assert "service" in response.json()

    def test_root_has_docs_link(self, client: TestClient) -> None:
        response = client.get("/")
        assert "docs" in response.json()

    def test_root_has_ops_links(self, client: TestClient) -> None:
        response = client.get("/")
        payload = response.json()
        assert payload["ops_resource_pack"] == "/api/v1/ops/resource-pack"
        assert payload["ops_release_readiness"] == "/api/v1/ops/release-readiness"


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["status"] == "ok"

    def test_health_has_version(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "version" in data

    def test_health_has_model_mode(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["model_mode"] in ("demo", "model")

    def test_health_has_defect_classes(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert isinstance(data["defect_classes"], list)
        assert len(data["defect_classes"]) == 7

    def test_health_has_review_routes(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["proof_routes"]["resource_pack"] == "/api/v1/ops/resource-pack"
        assert data["reviewer_fast_path"][1] == "/api/v1/ops/resource-pack"


class TestOpsSurfaces:
    def test_resource_pack_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/ops/resource-pack")
        assert response.status_code == 200
        data = response.json()
        assert data["contract_version"] == "weld-defect-review-resource-pack-v1"
        assert data["summary"]["defect_example_count"] >= 4

    def test_release_readiness_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/ops/release-readiness")
        assert response.status_code == 200
        data = response.json()
        assert data["contract_version"] == "weld-defect-release-readiness-v1"
        assert data["checks"]["inspection_api"] is True


class TestClassesEndpoint:
    def test_classes_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/classes")
        assert response.status_code == 200

    def test_classes_returns_all_types(self, client: TestClient) -> None:
        data = client.get("/api/v1/classes").json()
        assert len(data["classes"]) == 7

    def test_each_class_has_type_and_description(self, client: TestClient) -> None:
        data = client.get("/api/v1/classes").json()
        for cls in data["classes"]:
            assert "type" in cls
            assert "description" in cls
            assert len(cls["description"]) > 0


class TestInspectEndpoint:
    def _upload(self, client: TestClient, image: Image.Image, fmt: str = "PNG") -> dict:
        raw = image_to_bytes(image, fmt=fmt)
        response = client.post(
            "/api/v1/inspect",
            files={"file": (f"test.{fmt.lower()}", raw, f"image/{fmt.lower()}")},
        )
        return response

    def test_inspect_returns_200(self, client: TestClient) -> None:
        response = self._upload(client, make_image_no_defect())
        assert response.status_code == 200

    def test_inspect_response_has_report_id(self, client: TestClient) -> None:
        data = self._upload(client, make_image_no_defect()).json()
        assert "report_id" in data
        assert data["report_id"].startswith("WDV-")

    def test_inspect_detection_structure(self, client: TestClient) -> None:
        data = self._upload(client, make_image_porosity()).json()
        detection = data["detection"]
        assert "defect_type" in detection
        assert "confidence" in detection
        assert "is_defect" in detection
        assert "class_probabilities" in detection

    def test_inspect_severity_structure(self, client: TestClient) -> None:
        data = self._upload(client, make_image_crack()).json()
        severity = data["severity"]
        assert "score" in severity
        assert "level" in severity
        assert "recommended_action" in severity

    def test_inspect_confidence_valid_range(self, client: TestClient) -> None:
        data = self._upload(client, make_image_no_defect()).json()
        conf = data["detection"]["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_inspect_severity_score_valid_range(self, client: TestClient) -> None:
        data = self._upload(client, make_image_crack()).json()
        score = data["severity"]["score"]
        assert 0.0 <= score <= 100.0

    def test_inspect_conclusion_present(self, client: TestClient) -> None:
        data = self._upload(client, make_image_no_defect()).json()
        assert "conclusion" in data
        assert len(data["conclusion"]) > 0

    def test_inspect_jpeg_format(self, client: TestClient) -> None:
        response = self._upload(client, make_image_no_defect(), fmt="JPEG")
        assert response.status_code == 200

    def test_inspect_empty_file_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/inspect",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400

    def test_inspect_with_weld_joint_id(self, client: TestClient) -> None:
        raw = image_to_bytes(make_image_no_defect())
        response = client.post(
            "/api/v1/inspect",
            files={"file": ("test.png", raw, "image/png")},
            data={"weld_joint_id": "J-2024-001"},
        )
        data = response.json()
        assert data["weld_joint_id"] == "J-2024-001"

    def test_inspect_with_inspector_notes(self, client: TestClient) -> None:
        raw = image_to_bytes(make_image_no_defect())
        response = client.post(
            "/api/v1/inspect",
            files={"file": ("test.png", raw, "image/png")},
            data={"inspector_notes": "Visual inspection OK"},
        )
        assert response.status_code == 200

    def test_inspect_preprocessing_info_present(self, client: TestClient) -> None:
        data = self._upload(client, make_image_no_defect()).json()
        assert "preprocessing" in data
        assert isinstance(data["preprocessing"], dict)

    def test_inspect_unsupported_content_type_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/inspect",
            files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert response.status_code == 422


class TestInspectReportEndpoint:
    def test_report_returns_html(self, client: TestClient) -> None:
        raw = image_to_bytes(make_image_no_defect())
        response = client.post(
            "/api/v1/inspect/report",
            files={"file": ("test.png", raw, "image/png")},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<!DOCTYPE html>" in response.text

    def test_report_contains_report_id(self, client: TestClient) -> None:
        raw = image_to_bytes(make_image_no_defect())
        response = client.post(
            "/api/v1/inspect/report",
            files={"file": ("test.png", raw, "image/png")},
        )
        assert "WDV-" in response.text


class TestBatchInspectEndpoint:
    def test_batch_single_image(self, client: TestClient) -> None:
        raw = image_to_bytes(make_image_no_defect())
        response = client.post(
            "/api/v1/batch/inspect",
            files=[("files", ("img1.png", raw, "image/png"))],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["total"] == 1

    def test_batch_multiple_images(self, client: TestClient) -> None:
        files = [
            ("files", (f"img{i}.png", image_to_bytes(make_image_no_defect()), "image/png"))
            for i in range(3)
        ]
        response = client.post("/api/v1/batch/inspect", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["total"] == 3

    def test_batch_empty_returns_400(self, client: TestClient) -> None:
        response = client.post("/api/v1/batch/inspect", files=[])
        assert response.status_code in (400, 422)

    def test_batch_summary_has_pass_rate(self, client: TestClient) -> None:
        raw = image_to_bytes(make_image_no_defect())
        response = client.post(
            "/api/v1/batch/inspect",
            files=[("files", ("img.png", raw, "image/png"))],
        )
        assert "pass_rate" in response.json()["summary"]


class TestDemoEndpoint:
    def test_demo_synthetic_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/demo/synthetic")
        assert response.status_code == 200

    def test_demo_synthetic_has_report(self, client: TestClient) -> None:
        data = client.get("/api/v1/demo/synthetic").json()
        assert "report" in data
        assert "detection" in data["report"]
