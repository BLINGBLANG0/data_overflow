"""
Tests for Insurance Bundle Recommender API.

Run with:  pytest tests/ -v
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """Create test client with model loaded."""
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_policy():
    """A valid policy input dict."""
    return {
        "User_ID": "TEST_001",
        "Employment_Status": "Employed_FullTime",
        "Estimated_Annual_Income": 45000,
        "Region_Code": "ABJ",
        "Adult_Dependents": 2,
        "Child_Dependents": 1,
        "Infant_Dependents": 0,
        "Policy_Start_Year": 2016,
        "Policy_Start_Month": "March",
        "Policy_Start_Week": 12,
        "Policy_Start_Day": 15,
        "Grace_Period_Extensions": 0,
        "Previous_Policy_Duration_Months": 6,
        "Existing_Policyholder": 0,
        "Previous_Claims_Filed": 0,
        "Years_Without_Claims": 1,
        "Policy_Amendments_Count": 0,
        "Deductible_Tier": "Tier_2_Moderate_Ded",
        "Payment_Schedule": "Monthly_EFT",
        "Underwriting_Processing_Days": 3,
        "Vehicles_on_Policy": 1,
        "Custom_Riders_Requested": 0,
        "Days_Since_Quote": 5,
        "Acquisition_Channel": "Direct_Website",
        "Broker_Agency_Type": "National_Corporate",
        "Policy_Cancelled_Post_Purchase": 0,
        "Broker_ID": None,
        "Employer_ID": None,
    }


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_has_features(self, client):
        data = client.get("/health").json()
        assert data["num_features"] > 0

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data
        assert data["version"]

    def test_health_has_model_size(self, client):
        data = client.get("/health").json()
        assert data["model_size_mb"] > 0


# ---------------------------------------------------------------------------
# Single prediction endpoint
# ---------------------------------------------------------------------------
class TestPredictSingle:
    def test_predict_returns_200(self, client, sample_policy):
        resp = client.post("/predict", json=sample_policy)
        assert resp.status_code == 200

    def test_predict_has_required_fields(self, client, sample_policy):
        data = client.post("/predict", json=sample_policy).json()
        assert "User_ID" in data
        assert "predicted_bundle" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_predict_bundle_range(self, client, sample_policy):
        data = client.post("/predict", json=sample_policy).json()
        assert 0 <= data["predicted_bundle"] <= 9

    def test_predict_confidence_range(self, client, sample_policy):
        data = client.post("/predict", json=sample_policy).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, client, sample_policy):
        data = client.post("/predict", json=sample_policy).json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_user_id_preserved(self, client, sample_policy):
        data = client.post("/predict", json=sample_policy).json()
        assert data["User_ID"] == "TEST_001"

    def test_predict_with_minimal_fields(self, client):
        """Test prediction with sparse but valid input."""
        minimal = {
            "User_ID": "MINIMAL_001",
            "Employment_Status": "Employed_FullTime",
            "Estimated_Annual_Income": 20000,
            "Region_Code": "ABJ",
            "Adult_Dependents": 0,
            "Child_Dependents": 0,
            "Infant_Dependents": 0,
            "Policy_Start_Year": 2016,
            "Policy_Start_Month": "January",
            "Policy_Start_Week": 1,
            "Policy_Start_Day": 1,
            "Grace_Period_Extensions": 0,
            "Previous_Policy_Duration_Months": 0,
            "Existing_Policyholder": 0,
            "Previous_Claims_Filed": 0,
            "Years_Without_Claims": 0,
            "Policy_Amendments_Count": 0,
            "Deductible_Tier": "Tier_2_Moderate_Ded",
            "Payment_Schedule": "Monthly_EFT",
            "Underwriting_Processing_Days": 0,
            "Vehicles_on_Policy": 0,
            "Custom_Riders_Requested": 0,
            "Days_Since_Quote": 0,
            "Acquisition_Channel": "Direct_Website",
            "Broker_Agency_Type": "National_Corporate",
            "Policy_Cancelled_Post_Purchase": 0,
            "Broker_ID": None,
            "Employer_ID": None,
        }
        resp = client.post("/predict", json=minimal)
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["predicted_bundle"] <= 9

    def test_predict_rule9_trigger(self, client):
        """Policy matching class 9 rule should return bundle 9."""
        rule9_policy = {
            "User_ID": "RULE9_TEST",
            "Employment_Status": "Employed_FullTime",
            "Estimated_Annual_Income": 0,
            "Region_Code": None,
            "Adult_Dependents": 0,
            "Child_Dependents": 0,
            "Infant_Dependents": 0,
            "Policy_Start_Year": 2016,
            "Policy_Start_Month": "March",
            "Policy_Start_Week": 10,
            "Policy_Start_Day": 5,
            "Grace_Period_Extensions": 0,
            "Previous_Policy_Duration_Months": 0,
            "Existing_Policyholder": 0,
            "Previous_Claims_Filed": 0,
            "Years_Without_Claims": 0,
            "Policy_Amendments_Count": 0,
            "Deductible_Tier": "Tier_4_Zero_Ded",
            "Payment_Schedule": "Monthly_EFT",
            "Underwriting_Processing_Days": 0,
            "Vehicles_on_Policy": 0,
            "Custom_Riders_Requested": 0,
            "Days_Since_Quote": 0,
            "Acquisition_Channel": "Direct_Website",
            "Broker_Agency_Type": "National_Corporate",
            "Policy_Cancelled_Post_Purchase": 1,
            "Broker_ID": None,
            "Employer_ID": None,
        }
        data = client.post("/predict", json=rule9_policy).json()
        assert data["predicted_bundle"] == 9


# ---------------------------------------------------------------------------
# Batch prediction endpoint
# ---------------------------------------------------------------------------
class TestPredictBatch:
    def test_batch_returns_200(self, client, sample_policy):
        resp = client.post("/predict/batch", json={"policies": [sample_policy]})
        assert resp.status_code == 200

    def test_batch_count_matches(self, client, sample_policy):
        policies = [sample_policy.copy() for _ in range(5)]
        for i, p in enumerate(policies):
            p["User_ID"] = f"BATCH_{i}"
        data = client.post("/predict/batch", json={"policies": policies}).json()
        assert data["count"] == 5
        assert len(data["predictions"]) == 5

    def test_batch_empty_rejected(self, client):
        resp = client.post("/predict/batch", json={"policies": []})
        assert resp.status_code == 400

    def test_batch_has_model_version(self, client, sample_policy):
        data = client.post("/predict/batch", json={"policies": [sample_policy]}).json()
        assert "model_version" in data


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
class TestFrontend:
    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_static_css(self, client):
        resp = client.get("/static/style.css")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Swagger docs
# ---------------------------------------------------------------------------
class TestDocs:
    def test_openapi_schema(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "/predict" in data["paths"]
        assert "/health" in data["paths"]
