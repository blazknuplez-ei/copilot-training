"""Train the discount prediction model on synthetic data."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.discount_predictor import DiscountPredictor

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)


def load_synthetic_data(data_path: str = "data/synthetic_output/generated_data.json") -> pd.DataFrame:
    """Load and prepare synthetic data from JSON file.
    
    Note: Synth generates random IDs that don't match foreign keys, so we:
    1. Map passenger_id/route_id to indices in their respective arrays
    2. Use those indices to look up passenger/route data
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract the collections
    discounts = pd.DataFrame(data["discounts"])
    passengers = pd.DataFrame(data["passengers"])
    routes = pd.DataFrame(data["routes"])
    
    print(f"Loaded {len(discounts)} discounts, {len(passengers)} passengers, {len(routes)} routes")
    
    # Expand travel_history JSON into columns
    passengers["history_trips"] = passengers["travel_history"].apply(lambda x: x["trips"])
    passengers["total_spend"] = passengers["travel_history"].apply(lambda x: x["total_spend"])
    # Avoid division by zero
    passengers["avg_spend"] = passengers.apply(
        lambda row: row["total_spend"] / row["history_trips"] if row["history_trips"] > 0 else 0, 
        axis=1
    )
    
    # Map foreign keys to actual indices (workaround for Synth's random ID generation)
    # Use modulo to wrap around if foreign key exceeds array size
    discounts["passenger_idx"] = discounts["passenger_id"] % len(passengers)
    discounts["route_idx"] = discounts["route_id"] % len(routes)
    
    # Add passenger data using index-based lookup
    discounts["history_trips"] = discounts["passenger_idx"].map(
        lambda idx: passengers.iloc[idx]["history_trips"]
    )
    discounts["avg_spend"] = discounts["passenger_idx"].map(
        lambda idx: passengers.iloc[idx]["avg_spend"]
    )
    
    # Add route data using index-based lookup
    discounts["distance_km"] = discounts["route_idx"].map(
        lambda idx: routes.iloc[idx]["distance"]
    )
    discounts["origin"] = discounts["route_idx"].map(
        lambda idx: routes.iloc[idx]["origin"]
    )
    discounts["destination"] = discounts["route_idx"].map(
        lambda idx: routes.iloc[idx]["destination"]
    )
    
    print(f"Created dataset with {len(discounts)} records")
    
    # Keep only the required columns
    required_cols = ["discount_value", "distance_km", "history_trips", "avg_spend", 
                     "route_id", "origin", "destination"]
    df = discounts[required_cols]
    
    return df


def train_model(output_model_path: str = "models/discount_predictor.pkl") -> None:
    """Train the discount prediction model and save it."""
    print("Loading synthetic data...")
    df = load_synthetic_data()
    
    print(f"Loaded {len(df)} records")
    print(f"Features: {df.columns.tolist()}")
    print(f"\nData summary:")
    print(df.describe())
    
    # Separate features and target
    X = df[["distance_km", "history_trips", "avg_spend", "route_id", "origin", "destination"]]
    y = df["discount_value"]
    
    print(f"\nTraining model with {len(X)} samples...")
    
    # Create and train the model
    model = DiscountPredictor()
    model.fit(X, y)
    
    # Save the model
    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    
    print(f"\n✅ Model trained and saved to: {output_path}")
    print(f"Model is ready for predictions!")


if __name__ == "__main__":
    train_model()
