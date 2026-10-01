import os
import pytest
import numpy as np
import pandas as pd

from src.fraud_detection import (
    ensure_dataset,
    load_and_preprocess_data,
    calculate_reconstruction_error,
    predict_anomalies,
    FallbackIsolationForest,
    build_autoencoder_model,
)

def test_ensure_dataset(tmp_path):
    # Test dataset extraction logic
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create dummy csv and zip
    csv_file = data_dir / "creditcard.csv"
    csv_file.write_text("Time,V1,V2,Amount,Class\n0,1,1,100,0\n")

    result = ensure_dataset(data_dir=str(data_dir), csv_name="creditcard.csv", zip_name="creditcardfraud.zip")
    assert os.path.exists(result)

def test_calculate_reconstruction_error():
    X_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_pred = np.array([[1.0, 1.0], [1.0, 4.0]])
    # Expected squared diffs: [[0, 1], [4, 0]] -> means: [0.5, 2.0]
    errors = calculate_reconstruction_error(X_true, X_pred)
    np.testing.assert_allclose(errors, np.array([0.5, 2.0]))

def test_predict_anomalies():
    errors = np.array([1.0, 2.5, 3.0, 5.0])
    threshold = 2.9
    preds = predict_anomalies(errors, threshold=threshold)
    np.testing.assert_array_equal(preds, np.array([0, 0, 1, 1]))

def test_fallback_isolation_forest():
    np.random.seed(42)
    X = np.random.normal(size=(100, 5))
    model = FallbackIsolationForest()
    model.fit(X)
    preds = model.predict(X)
    assert len(preds) == 100
    assert set(np.unique(preds)).issubset({0, 1})

def test_build_autoencoder_model():
    model = build_autoencoder_model(input_dim=29, encoding_dim=14)
    assert model is not None
    assert len(model.layers) == 5
