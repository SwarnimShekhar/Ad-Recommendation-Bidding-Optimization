import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import os

def load_data(processed_data_path="data/processed_data.csv"):
    df = pd.read_csv(processed_data_path)
    return df

def prepare_features(df):
    # Define features and target (click prediction)
    X = df.drop(columns=["click"])
    y = df["click"]
    return X, y

def train_ctr_model():
    df = load_data()
    X, y = prepare_features(df)
    # For simplicity, use 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 42,
        "max_depth": 5,
        "eta": 0.1
    }
    evals = [(dtrain, "train"), (dtest, "eval")]
    model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10, verbose_eval=False)
    
    # Evaluate model
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred)
    print(f"CTR Prediction Model - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/ctr_model.xgb"
    model.save_model(model_path)
    print("Model saved to", model_path)
    return model

if __name__ == "__main__":
    train_ctr_model()