import os
import json
import joblib
import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier  # Use a better ML model

# Firebase RTDB URL
FIREBASE_ATTACKS_URL = (
    "https://iotsecurity-30d1a-default-rtdb.firebaseio.com/attacks.json"
)

# Flask App
app = Flask(__name__)

# Model and Label Encoder Paths
MODEL_PATH = "attack_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"


def fetch_firebase_data():
    """Fetch attack data from Firebase RTDB."""
    try:
        response = requests.get(FIREBASE_ATTACKS_URL)
        if response.status_code == 200:
            data = response.json()
            if data:
                return pd.DataFrame(data.values())  # Convert Firebase JSON to DataFrame
        print("⚠️ No attack data found.")
        return pd.DataFrame()  # Return empty DataFrame if no data
    except Exception as e:
        print(f"❌ Firebase fetch failed: {e}")
        return pd.DataFrame()


def preprocess_data(df):
    """Preprocess attack data for ML model training."""
    if df.empty:
        return None, None, None

    # Convert timestamp (handling ISO 8601 format)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)

    # Extract useful features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Encode attack categories
    label_encoder = LabelEncoder()
    df["attack_type"] = label_encoder.fit_transform(df["category"])

    X = df[["hour", "day_of_week"]]
    y = df["attack_type"]

    return X, y, label_encoder


def train_model():
    """Fetch data, train model, and save it."""
    df = fetch_firebase_data()
    X, y, label_encoder = preprocess_data(df)

    if X is None or y is None:
        print("⚠️ No valid data for training.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use XGBoost for better prediction
    model = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Save model and label encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print("✅ Model trained and saved successfully.")
    return model, label_encoder


# Load model if exists, else train a new one
if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✅ Loaded pre-trained model.")
else:
    print("⚠️ No pre-trained model found. Training a new one.")
    model, label_encoder = train_model()


@app.route("/predict", methods=["POST"])
def predict():
    """Predict attack type given timestamp."""
    data = request.get_json()
    timestamp = data.get("timestamp")

    if not timestamp:
        return jsonify({"error": "Timestamp required"}), 400

    try:
        ts = pd.to_datetime(timestamp, errors="coerce")
        if pd.isna(ts):
            return jsonify({"error": "Invalid timestamp format"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid timestamp: {e}"}), 400

    # ✅ Convert to DataFrame with correct feature names
    features = pd.DataFrame([[ts.hour, ts.dayofweek]], columns=["hour", "day_of_week"])
    prediction = model.predict(features)[0]

    try:
        attack_type = label_encoder.inverse_transform([prediction])[0]
    except:
        attack_type = "Unknown Attack"  # Handle unknown categories

    return jsonify(
        {"predicted_attack": attack_type, "hour": ts.hour, "day_of_week": ts.dayofweek}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
