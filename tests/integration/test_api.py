"""Integration tests for the Phase 1 API surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


class TestHealth:
    def test_returns_ok_with_versions(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__
        assert body["benchmark_version"]


class TestBenchmarks:
    def test_lists_benchmark_metadata(self, client):
        body = client.get("/benchmarks").json()
        assert body["case_count"] == 10
        assert len(body["case_types"]) == 10

    def test_lists_all_cases(self, client):
        body = client.get("/benchmarks/cases").json()
        assert len(body) == 10
        assert {"case_id", "case_type", "critical_fact_count"} <= body[0].keys()

    def test_returns_a_full_case_with_gold_standard_facts(self, client):
        response = client.get("/benchmarks/cases/CASE_002")
        assert response.status_code == 200
        body = response.json()
        assert body["case_id"] == "CASE_002"
        assert body["critical_facts"]
        assert body["temporal_facts"]

    def test_unknown_case_returns_404_with_message(self, client):
        response = client.get("/benchmarks/cases/CASE_DOES_NOT_EXIST")
        assert response.status_code == 404
        assert "CASE_DOES_NOT_EXIST" in response.json()["detail"]


class TestOpenAPI:
    def test_schema_is_generated(self, client):
        body = client.get("/openapi.json").json()
        assert body["info"]["title"] == "ClinEval AI"
        assert "/benchmarks/cases/{case_id}" in body["paths"]

    def test_description_states_research_only_scope(self, client):
        description = client.get("/openapi.json").json()["info"]["description"].lower()
        assert "not a medical device" in description
