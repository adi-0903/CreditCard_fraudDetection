import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

def ensure_dataset(data_dir="data", csv_name="creditcard.csv", zip_name="creditcardfraud.zip"):
    """
    Ensures that the creditcard dataset is available.
    If the CSV file does not exist, it extracts it from the ZIP archive.
    """
    csv_path = os.path.join(data_dir, csv_name)
    zip_path = os.path.join(data_dir, zip_name)

    if not os.path.exists(csv_path):
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        else:
            raise FileNotFoundError(f"Neither {csv_path} nor {zip_path} exists.")
    return csv_path

def load_and_preprocess_data(csv_path, random_seed=42):
    """
    Loads dataset, scales Amount column, drops Time column,
    and splits into train (normal transactions) and test sets.
    """
    df = pd.read_csv(csv_path)
    data = df.drop(columns=['Time'], errors='ignore').copy()

    if 'Amount' in data.columns:
        scaler = StandardScaler()
        data['Amount'] = scaler.fit_transform(data[['Amount']].values)

    X_train, X_test = train_test_split(data, test_size=0.2, random_state=random_seed)

    # Keep only normal transactions (Class == 0) for Autoencoder training
    X_train_normal = X_train[X_train['Class'] == 0].drop(columns=['Class']).values

    y_test = X_test['Class'].values
    X_test_all = X_test.drop(columns=['Class']).values

    return X_train_normal, X_test_all, y_test

def calculate_reconstruction_error(X_true, X_pred):
    """
    Calculates Mean Squared Error (MSE) reconstruction error efficiently using vectorized NumPy.
    """
    return np.mean((X_true - X_pred) ** 2, axis=1)

def predict_anomalies(reconstruction_errors, threshold=2.9):
    """
    Predicts anomalies (1 for fraud, 0 for normal) given reconstruction errors and a threshold.
    """
    return (np.asarray(reconstruction_errors) > threshold).astype(int)

class FallbackIsolationForest:
    """
    Scikit-learn Isolation Forest model as a fast, reliable baseline fallback when Keras/TF is unavailable.
    """
    def __init__(self, contamination=0.0017, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        # IsolationForest returns -1 for outliers (fraud) and 1 for inliers (normal)
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)

def build_autoencoder_model(input_dim, encoding_dim=14):
    """
    Builds a Keras Autoencoder model if Keras is installed.
    """
    try:
        from keras.models import Model
        from keras.layers import Input, Dense
        from keras import regularizers

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
        decoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
        decoder = Dense(input_dim, activation="relu")(decoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return autoencoder
    except ImportError as e:
        raise ImportError("Keras/TensorFlow is not available. Use FallbackIsolationForest instead.") from e
