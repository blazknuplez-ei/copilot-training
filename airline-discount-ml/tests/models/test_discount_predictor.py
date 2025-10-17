"""Tests for DiscountPredictor model.

This test suite validates:
- Model training (fit) and prediction (predict) workflows
- Input validation for DataFrames
- Handling of missing values and unknown categorical values  
- Index preservation during predictions
- Model persistence (save/load)
- Performance vs. baseline (DummyRegressor)
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.models.discount_predictor import DiscountPredictor


@pytest.fixture
def synthetic_data():
    """Generate synthetic training data with required features."""
    np.random.seed(42)
    
    n = 100
    
    X = pd.DataFrame({
        "distance_km": np.random.uniform(2000, 8000, n),
        "history_trips": np.random.randint(1, 50, n),
        "avg_spend": np.random.uniform(300, 1500, n),
        "route_id": [f"R{i % 10}" for i in range(n)],
        "origin": np.random.choice(["NYC", "LAX", "SFO", "ORD"], n),
        "destination": np.random.choice(["LON", "PAR", "TYO", "SYD"], n),
    })
    
    y = pd.Series(
        0.05 + 0.0001 * X["distance_km"] + 0.002 * X["history_trips"] + np.random.normal(0, 0.02, n),
        name="discount_value"
    )
    
    return X, y


def test_discount_predictor_fit_predict(synthetic_data):
    """Test DiscountPredictor can fit and predict."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    preds = model.predict(X)
    
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(X)
    assert preds.index.equals(X.index)
    assert preds.name == "discount_value"


def test_predict_before_fit_raises(synthetic_data):
    """Test predict raises error if called before fit."""
    X, _ = synthetic_data
    
    model = DiscountPredictor()
    
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)


def test_fit_validates_empty_X():
    """Test fit raises on empty DataFrame."""
    model = DiscountPredictor()
    
    with pytest.raises(ValueError, match="empty"):
        model.fit(pd.DataFrame(), pd.Series([1]))


def test_fit_validates_non_dataframe():
    """Test fit raises ValueError for non-DataFrame input."""
    model = DiscountPredictor()
    
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        model.fit([[1, 2, 3]], pd.Series([10]))


def test_fit_validates_none_input():
    """Test fit raises ValueError for None input."""
    model = DiscountPredictor()
    
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        model.fit(None, pd.Series([10]))


def test_fit_validates_series_input():
    """Test fit raises ValueError when Series passed instead of DataFrame."""
    model = DiscountPredictor()
    
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        model.fit(pd.Series([1, 2, 3]), pd.Series([10, 20, 30]))


def test_fit_validates_dict_input():
    """Test fit raises ValueError for dict input."""
    model = DiscountPredictor()
    
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        model.fit({"col": [1, 2]}, pd.Series([10, 20]))


def test_fit_validates_numpy_array_input():
    """Test fit raises ValueError for numpy array input."""
    model = DiscountPredictor()
    
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        model.fit(np.array([[1, 2, 3]]), pd.Series([10]))


def test_fit_validates_missing_columns(synthetic_data):
    """Test fit raises on missing required columns."""
    _, y = synthetic_data
    X_bad = pd.DataFrame({"distance_km": [3000]})
    
    model = DiscountPredictor()
    
    with pytest.raises(ValueError, match="column is not a column of the dataframe"):
        model.fit(X_bad, y.iloc[:1])


def test_save_load_roundtrip(synthetic_data):
    """Test save/load produces identical predictions."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    preds_before = model.predict(X)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.joblib"
        model.save(path)
        
        loaded = DiscountPredictor.load(path)
        preds_after = loaded.predict(X)
    
    pd.testing.assert_series_equal(preds_before, preds_after)


def test_model_outperforms_baseline(synthetic_data):
    """Test model beats DummyRegressor baseline on MAE and R²."""
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Baseline
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_r2 = r2_score(y_test, baseline_preds)
    
    # Model
    model = DiscountPredictor()
    model.fit(X_train, y_train)
    model_preds = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, model_preds)
    model_r2 = r2_score(y_test, model_preds)
    
    # Model should be better
    assert model_mae < baseline_mae, f"Model MAE {model_mae} should be < baseline {baseline_mae}"
    assert model_r2 > baseline_r2, f"Model R² {model_r2} should be > baseline {baseline_r2}"
    
    print(f"\nBaseline: MAE={baseline_mae:.3f}, R²={baseline_r2:.3f}")
    print(f"Model:    MAE={model_mae:.3f}, R²={model_r2:.3f}")


def test_predict_preserves_index():
    """Test predict preserves custom index."""
    X = pd.DataFrame({
        "distance_km": [3000, 4000],
        "history_trips": [5, 10],
        "avg_spend": [500, 800],
        "route_id": ["R1", "R2"],
        "origin": ["NYC", "LAX"],
        "destination": ["LON", "TYO"],
    }, index=["row_a", "row_b"])
    
    y = pd.Series([10.0, 15.0], index=["row_a", "row_b"])
    
    model = DiscountPredictor()
    model.fit(X, y)
    preds = model.predict(X)
    
    assert list(preds.index) == ["row_a", "row_b"]


# ===== Additional predict() method tests =====


def test_predict_returns_series_with_correct_name(synthetic_data):
    """Test predict returns Series with name 'discount_value'."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    preds = model.predict(X)
    
    assert preds.name == "discount_value"


def test_predict_validates_empty_dataframe(synthetic_data):
    """Test predict raises ValueError for empty DataFrame."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    X_empty = pd.DataFrame()
    
    with pytest.raises(ValueError, match="empty"):
        model.predict(X_empty)


def test_predict_validates_non_dataframe(synthetic_data):
    """Test predict raises ValueError for non-DataFrame input."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        model.predict([[3000, 5, 500]])


def test_predict_handles_missing_values(synthetic_data):
    """Test predict handles missing values via imputation."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    # Create test data with missing values
    X_test = X.iloc[:5].copy()
    X_test.loc[X_test.index[0], "distance_km"] = np.nan
    X_test.loc[X_test.index[1], "history_trips"] = np.nan
    X_test.loc[X_test.index[2], "route_id"] = np.nan
    
    # Should not raise - imputation should handle it
    preds = model.predict(X_test)
    
    assert len(preds) == 5
    assert not preds.isna().any(), "Predictions should not contain NaN"


def test_predict_handles_new_categorical_values(synthetic_data):
    """Test predict handles unseen categorical values."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    # Create test data with new categorical values
    X_test = pd.DataFrame({
        "distance_km": [3500],
        "history_trips": [8],
        "avg_spend": [650],
        "route_id": ["R999"],  # New value not in training
        "origin": ["ZZZ"],     # New value not in training
        "destination": ["WWW"], # New value not in training
    })
    
    # Should not raise - OneHotEncoder handles unknown with 'ignore'
    preds = model.predict(X_test)
    
    assert len(preds) == 1
    assert not preds.isna().any()


def test_predict_with_subset_of_training_data(synthetic_data):
    """Test predict works with subset of training data."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    # Predict on just first 10 rows
    X_subset = X.iloc[:10]
    preds = model.predict(X_subset)
    
    assert len(preds) == 10
    assert preds.index.equals(X_subset.index)


def test_predict_with_different_row_order(synthetic_data):
    """Test predict preserves index when rows are reordered."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    # Shuffle the data
    X_shuffled = X.sample(frac=1, random_state=123)
    preds = model.predict(X_shuffled)
    
    assert preds.index.equals(X_shuffled.index)
    assert len(preds) == len(X_shuffled)


def test_predict_single_row(synthetic_data):
    """Test predict works with single row DataFrame."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    X_single = X.iloc[[0]]
    preds = model.predict(X_single)
    
    assert len(preds) == 1
    assert isinstance(preds, pd.Series)
    assert preds.name == "discount_value"


def test_predict_returns_numeric_values(synthetic_data):
    """Test predict returns numeric predictions."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    preds = model.predict(X)
    
    assert pd.api.types.is_numeric_dtype(preds)
    assert not preds.isna().any()


def test_predict_consistency_multiple_calls(synthetic_data):
    """Test predict returns same results when called multiple times."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    preds1 = model.predict(X)
    preds2 = model.predict(X)
    
    pd.testing.assert_series_equal(preds1, preds2)


def test_predict_after_load_preserves_functionality(synthetic_data):
    """Test predict works correctly after save/load cycle."""
    X, y = synthetic_data
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train = y.iloc[:80]
    
    model = DiscountPredictor()
    model.fit(X_train, y_train)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.joblib"
        model.save(path)
        
        loaded_model = DiscountPredictor.load(path)
        preds = loaded_model.predict(X_test)
    
    assert len(preds) == len(X_test)
    assert preds.index.equals(X_test.index)
    assert preds.name == "discount_value"


def test_predict_with_extra_columns(synthetic_data):
    """Test predict ignores extra columns not in training."""
    X, y = synthetic_data
    
    model = DiscountPredictor()
    model.fit(X, y)
    
    # Add extra columns
    X_extra = X.copy()
    X_extra["extra_col"] = 999
    X_extra["another_col"] = "ignored"
    
    preds = model.predict(X_extra)
    
    # Should still work, ignoring extra columns
    assert len(preds) == len(X_extra)
    assert preds.index.equals(X_extra.index)


# ===== Direct _validate_X static method tests =====


def test_validate_X_accepts_valid_dataframe():
    """Test _validate_X accepts valid non-empty DataFrame."""
    X = pd.DataFrame({
        "distance_km": [3000, 4000],
        "history_trips": [5, 10],
        "avg_spend": [500, 800],
        "route_id": ["R1", "R2"],
        "origin": ["NYC", "LAX"],
        "destination": ["LON", "TYO"],
    })
    
    # Should not raise
    DiscountPredictor._validate_X(X)


def test_validate_X_accepts_dataframe_with_extra_columns():
    """Test _validate_X accepts DataFrame with extra columns."""
    X = pd.DataFrame({
        "col1": [1, 2],
        "col2": [3, 4],
        "extra": [5, 6]
    })
    
    # Should not raise (validation only checks type and emptiness)
    DiscountPredictor._validate_X(X)


def test_validate_X_accepts_dataframe_with_custom_index():
    """Test _validate_X accepts DataFrame with custom index."""
    X = pd.DataFrame({
        "col1": [1, 2],
        "col2": [3, 4]
    }, index=["row_a", "row_b"])
    
    # Should not raise
    DiscountPredictor._validate_X(X)


def test_validate_X_rejects_empty_dataframe():
    """Test _validate_X raises ValueError for empty DataFrame."""
    X_empty = pd.DataFrame()
    
    with pytest.raises(ValueError, match="empty"):
        DiscountPredictor._validate_X(X_empty)


def test_validate_X_rejects_dataframe_with_zero_rows():
    """Test _validate_X raises ValueError for DataFrame with columns but no rows."""
    X_zero_rows = pd.DataFrame(columns=["col1", "col2"])
    
    with pytest.raises(ValueError, match="empty"):
        DiscountPredictor._validate_X(X_zero_rows)


def test_validate_X_rejects_none():
    """Test _validate_X raises ValueError for None."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        DiscountPredictor._validate_X(None)


def test_validate_X_rejects_list():
    """Test _validate_X raises ValueError for list."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        DiscountPredictor._validate_X([[1, 2, 3]])


def test_validate_X_rejects_list_of_dicts():
    """Test _validate_X raises ValueError for list of dicts."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        DiscountPredictor._validate_X([{"col1": 1, "col2": 2}])


def test_validate_X_rejects_series():
    """Test _validate_X raises ValueError for Series."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        DiscountPredictor._validate_X(pd.Series([1, 2, 3]))


def test_validate_X_rejects_dict():
    """Test _validate_X raises ValueError for dict."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        DiscountPredictor._validate_X({"col1": [1, 2], "col2": [3, 4]})


def test_validate_X_rejects_numpy_array():
    """Test _validate_X raises ValueError for numpy array."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        DiscountPredictor._validate_X(np.array([[1, 2, 3], [4, 5, 6]]))


def test_validate_X_rejects_string():
    """Test _validate_X raises ValueError for string."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        DiscountPredictor._validate_X("not a dataframe")


def test_validate_X_rejects_integer():
    """Test _validate_X raises ValueError for integer."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        DiscountPredictor._validate_X(42)
