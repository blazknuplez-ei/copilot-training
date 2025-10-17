"""Tests for evaluation module."""
import pytest
from unittest.mock import MagicMock

from src.training.evaluate import evaluate_model, calculate_accuracy


class TestCalculateAccuracy:
    """Tests for accuracy calculation."""
    
    def test_calculate_accuracy_perfect_predictions(self):
        """Test accuracy with 100% correct predictions."""
        predictions = [1, 2, 3, 4, 5]
        labels = [1, 2, 3, 4, 5]
        
        accuracy = calculate_accuracy(predictions, labels)
        
        assert accuracy == 1.0
    
    def test_calculate_accuracy_half_correct(self):
        """Test accuracy with 50% correct predictions."""
        predictions = [1, 2, 3, 4, 5]
        labels = [1, 2, 0, 0, 5]
        
        accuracy = calculate_accuracy(predictions, labels)
        
        assert accuracy == 0.6  # 3 out of 5 correct
    
    def test_calculate_accuracy_no_correct(self):
        """Test accuracy with 0% correct predictions."""
        predictions = [1, 2, 3]
        labels = [4, 5, 6]
        
        accuracy = calculate_accuracy(predictions, labels)
        
        assert accuracy == 0.0
    
    def test_calculate_accuracy_empty_lists(self):
        """Test accuracy with empty lists."""
        accuracy = calculate_accuracy([], [])
        
        assert accuracy == 0.0
    
    def test_calculate_accuracy_single_prediction(self):
        """Test accuracy with single prediction."""
        accuracy = calculate_accuracy([1], [1])
        assert accuracy == 1.0
        
        accuracy = calculate_accuracy([1], [2])
        assert accuracy == 0.0


class TestEvaluateModel:
    """Tests for model evaluation."""
    
    def test_evaluate_model_returns_dict(self):
        """Test that evaluate_model returns a dictionary."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 2, 3]
        
        mock_test_data = MagicMock()
        mock_test_data.labels = [1, 2, 3]
        
        result = evaluate_model(mock_model, mock_test_data)
        
        assert isinstance(result, dict)
        assert 'predictions' in result
        assert 'accuracy' in result
    
    def test_evaluate_model_calls_predict(self):
        """Test that evaluate_model calls model.predict."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 2, 3]
        
        mock_test_data = MagicMock()
        mock_test_data.labels = [1, 2, 3]
        
        evaluate_model(mock_model, mock_test_data)
        
        mock_model.predict.assert_called_once_with(mock_test_data)
    
    def test_evaluate_model_returns_predictions(self):
        """Test that evaluate_model returns predictions."""
        mock_model = MagicMock()
        expected_predictions = [0.15, 0.25, 0.35]
        mock_model.predict.return_value = expected_predictions
        
        mock_test_data = MagicMock()
        mock_test_data.labels = [0.15, 0.25, 0.35]
        
        result = evaluate_model(mock_model, mock_test_data)
        
        assert result['predictions'] == expected_predictions
        assert result['accuracy'] == 1.0
    
    def test_evaluate_model_calculates_accuracy(self):
        """Test that evaluate_model calculates accuracy correctly."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 2, 3, 4, 5]
        
        mock_test_data = MagicMock()
        mock_test_data.labels = [1, 2, 0, 0, 5]
        
        result = evaluate_model(mock_model, mock_test_data)
        
        assert result['accuracy'] == 0.6  # 3 out of 5 correct
