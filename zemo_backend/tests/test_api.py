"""
API integration tests for all Zemo endpoints using FastAPI TestClient.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    """GET /health returns 200 with status=healthy."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "storage_path" in data


def test_upload_csv(client: TestClient, sample_csv_bytes: bytes) -> None:
    """POST /api/v1/datasets/upload with CSV file returns 200 with dataset_id."""
    resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("customers.csv", sample_csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "dataset_id" in data
    assert data["record_count"] == 20
    assert data["column_count"] == 5
    assert data["status"] == "parsed"


def test_upload_unsupported(client: TestClient) -> None:
    """POST with unsupported file extension (.txt) returns 400."""
    resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("notes.txt", b"plain text notes", "text/plain")},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["code"] == "UNSUPPORTED_FORMAT"


def test_list_datasets(client: TestClient, sample_csv_bytes: bytes) -> None:
    """Upload dataset then GET /api/v1/datasets to verify listing."""
    upload_resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("list_test.csv", sample_csv_bytes, "text/csv")},
    )
    assert upload_resp.status_code == 200
    dataset_id = upload_resp.json()["dataset_id"]

    list_resp = client.get("/api/v1/datasets")
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert isinstance(items, list)
    matching = [it for it in items if it["dataset_id"] == dataset_id]
    assert len(matching) == 1


def test_get_dataset(client: TestClient, sample_csv_bytes: bytes) -> None:
    """GET /api/v1/datasets/{id} returns metadata."""
    upload_resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("get_test.csv", sample_csv_bytes, "text/csv")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    get_resp = client.get(f"/api/v1/datasets/{dataset_id}")
    assert get_resp.status_code == 200
    meta = get_resp.json()
    assert meta["dataset_id"] == dataset_id
    assert meta["record_count"] == 20
    assert "email" in meta["columns"]


def test_get_nonexistent_dataset(client: TestClient) -> None:
    """GET /api/v1/datasets/fake-id returns 404."""
    resp = client.get("/api/v1/datasets/fake-id")
    assert resp.status_code == 404
    data = resp.json()
    assert data["code"] == "DATASET_NOT_FOUND"


def test_run_scan(client: TestClient, sample_csv_bytes: bytes) -> None:
    """POST /api/v1/scans with valid dataset_id returns ScanResult."""
    upload_resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("scan_test.csv", sample_csv_bytes, "text/csv")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    scan_resp = client.post("/api/v1/scans", json={"dataset_id": dataset_id})
    assert scan_resp.status_code == 200
    scan_data = scan_resp.json()
    assert scan_data["dataset_id"] == dataset_id
    assert scan_data["status"] == "completed"
    assert scan_data["summary"]["total"] > 0
    assert len(scan_data["stages"]) >= 3


def test_get_findings(client: TestClient, sample_csv_bytes: bytes) -> None:
    """GET /api/v1/datasets/{id}/findings after scan returns findings."""
    upload_resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("findings_test.csv", sample_csv_bytes, "text/csv")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    # Run scan first
    client.post("/api/v1/scans", json={"dataset_id": dataset_id})

    findings_resp = client.get(f"/api/v1/datasets/{dataset_id}/findings")
    assert findings_resp.status_code == 200
    findings_data = findings_resp.json()
    assert findings_data["dataset_id"] == dataset_id
    assert len(findings_data["findings"]) > 0
    assert findings_data["summary"]["total"] == len(findings_data["findings"])


def test_sanitize(client: TestClient, sample_csv_bytes: bytes) -> None:
    """POST /api/v1/sanitization/apply creates a new sanitized dataset."""
    upload_resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("sanitize_test.csv", sample_csv_bytes, "text/csv")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    # Scan to generate findings
    client.post("/api/v1/scans", json={"dataset_id": dataset_id})

    # Sanitize
    san_resp = client.post("/api/v1/sanitization/apply", json={"dataset_id": dataset_id})
    assert san_resp.status_code == 200
    san_data = san_resp.json()
    assert "sanitized_dataset_id" in san_data
    assert san_data["sanitized_dataset_id"] != dataset_id
    assert san_data["records_removed"] >= 1


def test_readiness(client: TestClient, sample_csv_bytes: bytes) -> None:
    """POST /api/v1/readiness/assess returns ReadinessReport."""
    upload_resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("readiness_test.csv", sample_csv_bytes, "text/csv")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    # Scan first
    client.post("/api/v1/scans", json={"dataset_id": dataset_id})

    read_resp = client.post("/api/v1/readiness/assess", json={"dataset_id": dataset_id})
    assert read_resp.status_code == 200
    read_data = read_resp.json()
    assert read_data["dataset_id"] == dataset_id
    assert "overall_status" in read_data
    assert "overall_score" in read_data
    assert len(read_data["checks"]) > 0


def test_delete_dataset(client: TestClient, sample_csv_bytes: bytes) -> None:
    """DELETE /api/v1/datasets/{id} returns 200 and removes the dataset."""
    upload_resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("delete_test.csv", sample_csv_bytes, "text/csv")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    del_resp = client.delete(f"/api/v1/datasets/{dataset_id}")
    assert del_resp.status_code == 200

    # Ensure it is now gone
    get_resp = client.get(f"/api/v1/datasets/{dataset_id}")
    assert get_resp.status_code == 404


def test_frontend_static_serving(client: TestClient) -> None:
    """Verify FastAPI serves index.html, js/api.js, and css/variables.css."""
    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "text/html" in index_resp.headers["content-type"]
    assert "Zemo" in index_resp.text

    api_js_resp = client.get("/js/api.js")
    assert api_js_resp.status_code == 200
    assert "/api/v1/datasets/upload" in api_js_resp.text

    css_resp = client.get("/css/variables.css")
    assert css_resp.status_code == 200


def test_models_and_workflow_endpoints(client: TestClient) -> None:
    """Verify model catalog, config save, sample dataset, risk, validation, and workflow routes."""
    prov_resp = client.get("/api/v1/models/providers")
    assert prov_resp.status_code == 200
    assert len(prov_resp.json()["providers"]) == 6

    cfg_resp = client.post(
        "/api/v1/models/config",
        json={
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "environment": "Cloud Isolated",
            "task": "Supervised Fine-Tuning",
            "privacy_level": "Strict",
        },
    )
    assert cfg_resp.status_code == 200
    assert cfg_resp.json()["config"]["provider"] == "anthropic"

    sample_resp = client.post("/api/v1/datasets/sample")
    assert sample_resp.status_code == 200
    dataset_id = sample_resp.json()["dataset_id"]

    scan_resp = client.post("/api/v1/scans", json={"dataset_id": dataset_id})
    assert scan_resp.status_code == 200

    risk_resp = client.get(f"/api/v1/datasets/{dataset_id}/risk")
    assert risk_resp.status_code == 200
    assert "risk" in risk_resp.json()
    assert "policy" in risk_resp.json()

    wf_resp = client.get(f"/api/v1/workflow/{dataset_id}")
    assert wf_resp.status_code == 200
    assert len(wf_resp.json()["pipelineStages"]) == 7

    san_resp = client.post("/api/v1/sanitization/apply", json={"dataset_id": dataset_id})
    assert san_resp.status_code == 200

    val_resp = client.post(f"/api/v1/sanitization/{dataset_id}/validate")
    assert val_resp.status_code == 200
    assert val_resp.json()["passed"] is True


def test_lan_configuration_and_cors(client: TestClient) -> None:
    """Verify LAN host binding (0.0.0.0), LAN IP detection, RFC1918 CORS acceptance, and public origin rejection."""
    from app.core.config import get_lan_ips, get_primary_lan_ip, get_settings

    settings = get_settings()
    assert settings.host == "0.0.0.0"

    lan_ips = get_lan_ips()
    assert isinstance(lan_ips, list) and len(lan_ips) >= 1
    assert get_primary_lan_ip() == lan_ips[0]

    # 1. Verify RFC 1918 LAN origins (192.168/16, 172.16/12, 10/8, .local) are allowed by CORS
    for lan_origin in (
        "http://192.168.1.10:3000",
        "http://172.16.9.195:8000",
        "http://10.0.0.42:5500",
        "http://zemo-host.local:8000",
        "http://localhost:8000",
    ):
        preflight = client.options(
            "/api/v1/datasets",
            headers={
                "Origin": lan_origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers.get("access-control-allow-origin") == lan_origin

    # 2. Verify arbitrary external public origins are rejected by CORS
    disallowed = client.options(
        "/api/v1/datasets",
        headers={
            "Origin": "https://evil-attacker.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert disallowed.headers.get("access-control-allow-origin") is None

    # 3. Verify frontend js/api.js uses dynamic LAN hostname resolution rather than hardcoded 127.0.0.1
    api_js_resp = client.get("/js/api.js")
    assert api_js_resp.status_code == 200
    assert "getDynamicBackendOrigin" in api_js_resp.text
    assert 'const DEFAULT_BACKEND_ORIGIN = "http://127.0.0.1:8000"' not in api_js_resp.text


