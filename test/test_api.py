from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# Test 1: Home endpoint
def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Employee Prediction API is running"
    )


# Test 2: Health endpoint
def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# Test 3: Valid prediction request
def test_valid_prediction():
    payload = {
        "TotalMonthsOfExp": 60,
        "TotalOrgsWorked": 3,
        "MonthsInOrg": 24,
        "LastPayIncrementBand": 2,
        "AverageFeedback": 3
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["predicted_class"], int)
    assert "class_probabilities" in data
    assert data["model_version"] == "1.1.0"

    probabilities = data["class_probabilities"]

    assert probabilities
    assert all(
        0 <= probability <= 1
        for probability in probabilities.values()
    )

    assert abs(sum(probabilities.values()) - 1.0) < 1e-6


# Test 4: Reject negative experience
def test_negative_experience():
    payload = {
        "TotalMonthsOfExp": -5,
        "TotalOrgsWorked": 3,
        "MonthsInOrg": 24,
        "LastPayIncrementBand": 2,
        "AverageFeedback": 3
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# Test 5: Reject a missing field
def test_missing_field():
    payload = {
        "TotalMonthsOfExp": 60,
        "TotalOrgsWorked": 3,
        "MonthsInOrg": 24,
        "LastPayIncrementBand": 2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# Test 6: Unknown endpoint
def test_unknown_endpoint():
    response = client.get("/unknown")

    assert response.status_code == 404