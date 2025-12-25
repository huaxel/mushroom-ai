from fastapi.testclient import TestClient
from api.app import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid_input():
    # Example valid input based on usage instructions
    valid_data = {
        "cap-diameter": 5.0,
        "cap-shape": "b",
        "cap-surface": "s",
        "cap-color": "n",
        "does-bruise-or-bleed": "f",
        "gill-attachment": "a",
        "gill-spacing": "c",
        "gill-color": "n",
        "stem-height": 7.0,
        "stem-width": 1.2,
        "stem-color": "n",
        "has-ring": "t",
        "ring-type": "c",
        "habitat": "g",
        "season": "s"
    }
    
    response = client.post("/predict", json=valid_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)
    assert response.json()["prediction"] in [0, 1]

def test_predict_missing_field():
    # Missing 'season'
    invalid_data = {
        "cap-diameter": 5.0,
        "cap-shape": "b"
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422 # Pydantic validation error

def test_predict_invalid_type():
    # 'cap-diameter' should be float/int, sending string that can't be parsed easily if strict
    # Pydantic attempts coercion, but let's try something completely wrong for a float field if possible,
    # or a string field.
    # Actually, Pydantic 2.x is stricter.
    
    valid_data = {
        "cap-diameter": "not-a-number",
        "cap-shape": "b",
        "cap-surface": "s",
        "cap-color": "n",
        "does-bruise-or-bleed": "f",
        "gill-attachment": "a",
        "gill-spacing": "c",
        "gill-color": "n",
        "stem-height": 7.0,
        "stem-width": 1.2,
        "stem-color": "n",
        "has-ring": "t",
        "ring-type": "c",
        "habitat": "g",
        "season": "s"
    }
    
    response = client.post("/predict", json=valid_data)
    assert response.status_code == 422
