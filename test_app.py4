import pytest
from fastapi.testclient import TestClient
from app import app, model
import numpy as np

# Create test client
client = TestClient(app)


class TestPredictEndpoint:
    """Test the /predict API endpoint"""
    
    def test_predict_single_sample(self):
        """Test prediction with a single sample"""
        response = client.post(
            "/predict",
            json={"data": [5.1, 3.5, 1.4, 0.2]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 1
        assert data["predictions"][0] in [0, 1, 2]  # Valid iris classes
    
    def test_predict_multiple_samples(self):
        """Test prediction with multiple samples"""
        response = client.post(
            "/predict",
            json={"data": [
                [5.1, 3.5, 1.4, 0.2],
                [6.2, 2.9, 4.3, 1.3],
                [7.3, 2.9, 6.3, 1.8]
            ]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        for pred in data["predictions"]:
            assert pred in [0, 1, 2]
    
    def test_predict_invalid_data(self):
        """Test with invalid data format"""
        response = client.post(
            "/predict",
            json={"data": "invalid"}
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_missing_data(self):
        """Test with missing data field"""
        response = client.post(
            "/predict",
            json={}
        )
        assert response.status_code == 422
    
    def test_predict_wrong_feature_count(self):
        """Test with wrong number of features"""
        response = client.post(
            "/predict",
            json={"data": [5.1, 3.5]}  # Only 2 features instead of 4
        )
        # Should return 500 or handle gracefully
        assert response.status_code in [422, 500]


class TestWebInterface:
    """Test the web form interface"""
    
    def test_form_get(self):
        """Test GET request to form page"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Iris Prediction" in response.content
    
    def test_form_post_valid(self):
        """Test POST with valid form data"""
        response = client.post(
            "/",
            data={
                "f1": "5.1",
                "f2": "3.5",
                "f3": "1.4",
                "f4": "0.2"
            }
        )
        assert response.status_code == 200
        assert b"Response from /predict" in response.content
        assert b"predictions" in response.content
    
    def test_form_post_different_values(self):
        """Test POST with different iris measurements"""
        # Versicolor-like measurements
        response = client.post(
            "/",
            data={
                "f1": "6.2",
                "f2": "2.9",
                "f3": "4.3",
                "f4": "1.3"
            }
        )
        assert response.status_code == 200
        assert b"status_code" in response.content
    
    def test_form_post_missing_field(self):
        """Test POST with missing form field"""
        response = client.post(
            "/",
            data={
                "f1": "5.1",
                "f2": "3.5",
                "f3": "1.4"
                # f4 is missing
            }
        )
        assert response.status_code == 422


class TestModel:
    """Test the ML model"""
    
    def test_model_exists(self):
        """Test that model is loaded"""
        assert model is not None
    
    def test_model_can_predict(self):
        """Test model can make predictions"""
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = model.predict(sample)
        assert len(prediction) == 1
        assert prediction[0] in [0, 1, 2]
    
    def test_model_predictions_consistent(self):
        """Test model gives consistent predictions"""
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        pred1 = model.predict(sample)[0]
        pred2 = model.predict(sample)[0]
        assert pred1 == pred2  # Same input should give same output


class TestAppConfiguration:
    """Test app configuration"""
    
    def test_app_title(self):
        """Test app has correct title"""
        assert app.title == "Iris RF Demo"
    
    def test_root_endpoint_exists(self):
        """Test root endpoint is accessible"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_predict_endpoint_exists(self):
        """Test predict endpoint is accessible"""
        response = client.post(
            "/predict",
            json={"data": [5.1, 3.5, 1.4, 0.2]}
        )
        assert response.status_code == 200


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_extreme_values(self):
        """Test with extreme but valid values"""
        response = client.post(
            "/predict",
            json={"data": [10.0, 10.0, 10.0, 10.0]}
        )
        assert response.status_code == 200
    
    def test_zero_values(self):
        """Test with zero values"""
        response = client.post(
            "/predict",
            json={"data": [0.0, 0.0, 0.0, 0.0]}
        )
        assert response.status_code == 200
    
    def test_negative_values(self):
        """Test with negative values (unusual but should handle)"""
        response = client.post(
            "/predict",
            json={"data": [-1.0, -1.0, -1.0, -1.0]}
        )
        assert response.status_code == 200


# Run all tests with: pytest test_app.py -v
# Run specific test class: pytest test_app.py::TestPredictEndpoint -v
# Run with coverage: pytest test_app.py --cov=app --cov-report=html