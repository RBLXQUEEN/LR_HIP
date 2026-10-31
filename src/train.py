"""
Health Insurance Cost Prediction
================================
Multiple linear regression with a smoker x BMI interaction, fitted by OLS.

Predicts annual medical costs billed to health insurance from six policyholder
attributes.

Usage
-----
    python src/train.py

Reads   data/insurance.csv
Writes  model/coefficients.json   (consumed by index.html)

Course: Predictive Analytics (CSE 3141), Manipal University Jaipur
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "insurance.csv"
OUT = ROOT / "model" / "coefficients.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURES = ["age", "bmi", "children", "male", "smoker",
            "nw", "se", "sw", "smoker_x_bmi"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categoricals and add the interaction term.

    `northeast` is the baseline region, so it is represented by all three
    region dummies being zero.
    """
    X = pd.DataFrame({
        "age": df["age"].astype(float),
        "bmi": df["bmi"].astype(float),
        "children": df["children"].astype(float),
        "male": (df["sex"] == "male").astype(float),
        "smoker": (df["smoker"] == "yes").astype(float),
        "nw": (df["region"] == "northwest").astype(float),
        "se": (df["region"] == "southeast").astype(float),
        "sw": (df["region"] == "southwest").astype(float),
    })
    # The term that captures "the cost of smoking compounds with body mass".
    X["smoker_x_bmi"] = X["smoker"] * X["bmi"]
    return X[FEATURES]


def evaluate(y_true, y_pred, label: str) -> dict:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"{label:<22} R2={r2:.3f}  MAE=${mae:,.0f}  RMSE=${rmse:,.0f}")
    return {"r2": float(r2), "mae": float(mae), "rmse": rmse}


def main() -> None:
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} records, {df.shape[1]} columns")
    print(f"Missing values: {int(df.isna().sum().sum())}\n")

    X = build_features(df)
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train {len(X_train):,} rows / Test {len(X_test):,} rows\n")

    # --- benchmark: same features, no interaction -------------------------
    plain = LinearRegression().fit(X_train.drop(columns=["smoker_x_bmi"]),
                                   y_train)
    plain_pred = plain.predict(X_test.drop(columns=["smoker_x_bmi"]))
    print("Hold-out comparison")
    evaluate(y_test, plain_pred, "linear (no interaction)")

    # --- the model that is actually shipped ------------------------------
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = evaluate(y_test, pred, "linear + smoker x BMI")

    cv = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").mean()
    print(f"{'5-fold CV R2':<22} {cv:.3f}\n")

    # --- refit on everything for the deployed coefficients ----------------
    final = LinearRegression().fit(X, y)
    print(f"Intercept: ${final.intercept_:,.2f}")
    for name, coef in zip(FEATURES, final.coef_):
        print(f"  {name:<14} {coef:+,.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "intercept": float(final.intercept_),
        "coef": {f: float(c) for f, c in zip(FEATURES, final.coef_)},
        "metrics": metrics,
        "cv_r2": float(cv),
        "n": int(len(df)),
        "avg_charge": float(y.mean()),
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
