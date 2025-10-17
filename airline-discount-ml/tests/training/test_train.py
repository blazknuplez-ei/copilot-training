"""Tests for training module."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.training.train import load_synthetic_data, train_model


@pytest.fixture
def sample_synthetic_data():
    """Create sample synthetic data structure for testing."""
    return {
        "discounts": [
            {"passenger_id": 1, "route_id": 1, "discount_value": 0.15},
            {"passenger_id": 2, "route_id": 2, "discount_value": 0.25},
            {"passenger_id": 3, "route_id": 1, "discount_value": 0.20},
        ],
        "passengers": [
            {"travel_history": {"trips": 10, "total_spend": 5000}},
            {"travel_history": {"trips": 5, "total_spend": 2000}},
            {"travel_history": {"trips": 15, "total_spend": 8000}},
        ],
        "routes": [
            {"distance": 3000, "origin": "NYC", "destination": "LON"},
            {"distance": 5000, "origin": "LAX", "destination": "TYO"},
        ]
    }


class TestLoadSyntheticData:
    """Tests for loading synthetic data."""
    
    def test_load_synthetic_data_returns_dataframe(self, sample_synthetic_data, tmp_path):
        """Test that load_synthetic_data returns a DataFrame."""
        # Create temporary JSON file
        data_file = tmp_path / "generated_data.json"
        with open(data_file, "w") as f:
            json.dump(sample_synthetic_data, f)
        
        df = load_synthetic_data(str(data_file))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
    
    def test_load_synthetic_data_has_required_columns(self, sample_synthetic_data, tmp_path):
        """Test that loaded data has all required columns."""
        data_file = tmp_path / "generated_data.json"
        with open(data_file, "w") as f:
            json.dump(sample_synthetic_data, f)
        
        df = load_synthetic_data(str(data_file))
        
        required_cols = ["discount_value", "distance_km", "history_trips", 
                        "avg_spend", "route_id", "origin", "destination"]
        assert all(col in df.columns for col in required_cols)
    
    def test_load_synthetic_data_calculates_avg_spend(self, sample_synthetic_data, tmp_path):
        """Test that avg_spend is calculated correctly."""
        data_file = tmp_path / "generated_data.json"
        with open(data_file, "w") as f:
            json.dump(sample_synthetic_data, f)
        
        df = load_synthetic_data(str(data_file))
        
        # With modulo mapping: discount[0] -> passenger_idx=1%3=1 -> passengers[1]
        # passengers[1] has trips=5, total_spend=2000 -> avg_spend=400
        assert df.iloc[0]["avg_spend"] == 400.0
    
    def test_load_synthetic_data_handles_zero_trips(self, tmp_path):
        """Test that zero trips doesn't cause division by zero."""
        data = {
            "discounts": [{"passenger_id": 0, "route_id": 0, "discount_value": 0.10}],
            "passengers": [{"travel_history": {"trips": 0, "total_spend": 0}}],
            "routes": [{"distance": 2000, "origin": "NYC", "destination": "LAX"}]
        }
        data_file = tmp_path / "generated_data.json"
        with open(data_file, "w") as f:
            json.dump(data, f)
        
        df = load_synthetic_data(str(data_file))
        
        assert df.iloc[0]["avg_spend"] == 0.0


class TestTrainModel:
    """Tests for model training."""
    
    @patch('src.training.train.load_synthetic_data')
    @patch('src.training.train.DiscountPredictor')
    def test_train_model_loads_data(self, mock_predictor, mock_load_data, tmp_path):
        """Test that train_model loads synthetic data."""
        # Setup mock data
        mock_df = pd.DataFrame({
            "discount_value": [0.15, 0.25],
            "distance_km": [3000, 5000],
            "history_trips": [10, 5],
            "avg_spend": [500, 400],
            "route_id": [1, 2],
            "origin": ["NYC", "LAX"],
            "destination": ["LON", "TYO"]
        })
        mock_load_data.return_value = mock_df
        
        # Setup mock model
        mock_model_instance = MagicMock()
        mock_predictor.return_value = mock_model_instance
        
        output_path = tmp_path / "test_model.pkl"
        train_model(str(output_path))
        
        mock_load_data.assert_called_once()
    
    @patch('src.training.train.load_synthetic_data')
    @patch('src.training.train.DiscountPredictor')
    def test_train_model_fits_predictor(self, mock_predictor, mock_load_data, tmp_path):
        """Test that train_model fits the DiscountPredictor."""
        mock_df = pd.DataFrame({
            "discount_value": [0.15, 0.25],
            "distance_km": [3000, 5000],
            "history_trips": [10, 5],
            "avg_spend": [500, 400],
            "route_id": [1, 2],
            "origin": ["NYC", "LAX"],
            "destination": ["LON", "TYO"]
        })
        mock_load_data.return_value = mock_df
        
        mock_model_instance = MagicMock()
        mock_predictor.return_value = mock_model_instance
        
        output_path = tmp_path / "test_model.pkl"
        train_model(str(output_path))
        
        mock_model_instance.fit.assert_called_once()
    
    @patch('src.training.train.load_synthetic_data')
    @patch('src.training.train.DiscountPredictor')
    def test_train_model_saves_model(self, mock_predictor, mock_load_data, tmp_path):
        """Test that train_model saves the trained model."""
        mock_df = pd.DataFrame({
            "discount_value": [0.15, 0.25],
            "distance_km": [3000, 5000],
            "history_trips": [10, 5],
            "avg_spend": [500, 400],
            "route_id": [1, 2],
            "origin": ["NYC", "LAX"],
            "destination": ["LON", "TYO"]
        })
        mock_load_data.return_value = mock_df
        
        mock_model_instance = MagicMock()
        mock_predictor.return_value = mock_model_instance
        
        output_path = tmp_path / "test_model.pkl"
        train_model(str(output_path))
        
        mock_model_instance.save.assert_called_once()
        # Verify the path passed to save is correct
        save_call_path = mock_model_instance.save.call_args[0][0]
        assert str(output_path) in save_call_path
