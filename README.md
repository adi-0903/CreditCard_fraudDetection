# Credit Card Fraud Detection using Autoencoders in Keras

This project trains an Autoencoder Neural Network (in Keras/TensorFlow) for unsupervised/semi-supervised anomaly detection on anonymized credit card transaction data.

## Requirements & Setup

1. **Install Dependencies**
   Ensure Python 3.8+ is installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Extract Dataset**
   The dataset `creditcard.csv` is contained in `data/creditcardfraud.zip`. Extract it before running:
   ```bash
   unzip data/creditcardfraud.zip -d data/
   ```

3. **Run Notebook**
   Launch Jupyter Notebook and open `fraud_detection.ipynb`:
   ```bash
   jupyter notebook fraud_detection.ipynb
   ```
   Or execute the notebook non-interactively:
   ```bash
   jupyter nbconvert --to notebook --execute fraud_detection.ipynb --output executed_fraud_detection.ipynb
   ```

## Key Fixes & Modern Python Compatibility Notes
- **Pandas Compatibility**: Fixed `pd.value_counts(...)` (deprecated in modern Pandas) to `df['Class'].value_counts(...)`.
- **Keras 3 / TensorFlow 2 Compatibility**: Updated `load_model('model.h5')` to `load_model('model.h5', compile=False)` to prevent optimizer keyword argument mismatch errors when loading pre-trained models.
- **Dataset**: Provided dataset in `data/creditcard.csv` (unzipped from `data/creditcardfraud.zip`).
