"""
solution.py — Ship or It Didn’t Happen

Optional API:
    uvicorn solution:app --reload
"""
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("model.joblib")

def load_data():
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target

def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ])

def train_and_save_model(path=MODEL_PATH):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Evaluation report:")
    print(classification_report(y_test, y_pred))
    joblib.dump({"model": pipeline, "feature_names": list(X.columns)}, path)
    print(f"Saved model artefact to {path}")
    return pipeline

def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run train_and_save_model() first.")
    artefact = joblib.load(path)
    return artefact["model"], artefact["feature_names"]

def predict_one(features: List[float]):
    model, feature_names = load_model()
    if len(features) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} features, received {len(features)}.")
    row = pd.DataFrame([features], columns=feature_names)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][prediction])
    return {"prediction": prediction, "probability": probability}

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    app = FastAPI(title="ML Kata Prediction API")

    class PredictionRequest(BaseModel):
        features: List[float]

    @app.get("/")
    def root():
        return {"message": "ML kata model API is running"}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        return predict_one(request.features)
except ImportError:
    app = None

def main():
    train_and_save_model()
    X, _ = load_data()
    sample = X.iloc[0].tolist()
    print("\nSample prediction:")
    print(predict_one(sample))

if __name__ == "__main__":
    main()
