"""Tests for config module."""
import pytest

from src.utils import config


class TestConfig:
    """Tests for configuration functionality."""
    
    def test_database_url_is_string(self):
        """Test that DATABASE_URL is a string."""
        assert isinstance(config.DATABASE_URL, str)
        assert len(config.DATABASE_URL) > 0
    
    def test_model_path_is_string(self):
        """Test that MODEL_PATH is a string."""
        assert isinstance(config.MODEL_PATH, str)
        assert config.MODEL_PATH.endswith('.pkl')
    
    def test_log_level_is_valid(self):
        """Test that LOG_LEVEL is a valid level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.LOG_LEVEL in valid_levels
    
    def test_discount_threshold_is_float(self):
        """Test that DISCOUNT_THRESHOLD is a float."""
        assert isinstance(config.DISCOUNT_THRESHOLD, float)
        assert 0.0 <= config.DISCOUNT_THRESHOLD <= 1.0
    
    def test_features_is_list(self):
        """Test that FEATURES is a list."""
        assert isinstance(config.FEATURES, list)
        assert len(config.FEATURES) > 0
    
    def test_features_contains_strings(self):
        """Test that FEATURES contains only strings."""
        assert all(isinstance(feature, str) for feature in config.FEATURES)
    
    def test_database_url_format(self):
        """Test that DATABASE_URL follows expected format."""
        # Should contain postgresql protocol
        assert "postgresql://" in config.DATABASE_URL or "sqlite://" in config.DATABASE_URL
    
    def test_model_path_directory(self):
        """Test that MODEL_PATH is in models directory."""
        assert "models/" in config.MODEL_PATH
