"""End-to-end tests for analyze endpoint."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.skip(reason="Requires running service with mock backend")
class TestAnalyzeEndpoint:
    """E2E tests for /analyze endpoint."""

    def test_analyze_simple_email(self, test_client: TestClient) -> None:
        """Test analyzing a simple email."""
        response = test_client.post(
            "/analyze",
            json={"content": "Hello, this is a test email about invoices."},
        )

        assert response.status_code == 200
        data = response.json()

        assert "analysis" in data
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0

        analysis = data["analysis"]
        assert "short_summary" in analysis
        assert "detailed_summary" in analysis
        assert "keywords" in analysis
        assert "categories" in analysis
        assert "tone" in analysis
        assert "detected_language" in analysis

    def test_analyze_empty_content_rejected(self, test_client: TestClient) -> None:
        """Test that empty content is rejected."""
        response = test_client.post("/analyze", json={"content": ""})

        assert response.status_code == 422

    def test_analyze_missing_content_rejected(self, test_client: TestClient) -> None:
        """Test that missing content is rejected."""
        response = test_client.post("/analyze", json={})

        assert response.status_code == 422

    def test_health_endpoint(self, test_client: TestClient) -> None:
        """Test health endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready_endpoint(self, test_client: TestClient) -> None:
        """Test readiness endpoint."""
        response = test_client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "details" in data

    def test_stats_endpoint(self, test_client: TestClient) -> None:
        """Test stats endpoint."""
        response = test_client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert "batch_processor" in data
