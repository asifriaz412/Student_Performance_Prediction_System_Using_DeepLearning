"""
Fast training script — produces all artifacts required by app.py (REQUIRED_FILES).
Trains classical ML models only (no Keras) for quick FYP setup.
Run:  python train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_academic_behavior_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_RISK = "risk_status"
TARGET_GRADE = "final_grade"
TARGET_GPA = "final_gpa"

NUMERIC_COLS = [
    "hours_studied", "attendance", "previous_gpa",
    "tutoring_sessions", "sleep_hours", "stress_level",
    "screen_time", "exam_anxiety", "age",
]
CATEGORICAL_COLS = [
    "gender", "part_time_job", "study_method",
    "diet_quality", "internet_quality", "extracurricular",
    "family_income",
]
ENG_COLS = [
    "study_per_sleep", "stress_screen_sum", "anxiety_stress_ratio",
    "attendance_study_product", "cumulative_risk_score", "study_efficiency",
]
ALL_NUMERIC = NUMERIC_COLS + ENG_COLS
RANDOM_STATE = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["study_per_sleep"] = df["hours_studied"] / (df["sleep_hours"] + 1e-6)
    df["stress_screen_sum"] = df["stress_level"] + df["screen_time"]
    df["anxiety_stress_ratio"] = df["exam_anxiety"] / (df["stress_level"] + 1e-6)
    df["attendance_study_product"] = df["attendance"] * df["hours_studied"] / 100.0
    df["cumulative_risk_score"] = (
        df["stress_level"] + df["exam_anxiety"] - df["sleep_hours"]
    ).clip(lower=0)
    df["study_efficiency"] = df["hours_studied"] / (df["screen_time"] + 1.0)
    return df


def build_preprocessor() -> ColumnTransformer:
    num_tf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", RobustScaler()),
    ])
    cat_tf = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_tf, ALL_NUMERIC),
        ("cat", cat_tf, CATEGORICAL_COLS),
    ])


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = normalize_categoricals(df)
    df = engineer_features(df)

    risk_le = LabelEncoder()
    y_risk = risk_le.fit_transform(df[TARGET_RISK].astype(str))
    grade_le = LabelEncoder()
    y_grade = grade_le.fit_transform(df[TARGET_GRADE].astype(str))
    y_gpa = df[TARGET_GPA].values.astype(np.float32)

    X = df[ALL_NUMERIC + CATEGORICAL_COLS]
    (
        X_train, X_test,
        yr_tr, yr_te,
        yg_tr, yg_te,
        ygpa_tr, ygpa_te,
    ) = train_test_split(
        X, y_risk, y_grade, y_gpa,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_risk,
    )

    prep = build_preprocessor()
    prep_nn = build_preprocessor()
    X_tr = prep.fit_transform(X_train)
    X_te = prep.transform(X_test)
    prep_nn.fit(X_train)

    joblib.dump(risk_le, MODEL_DIR / "risk_encoder.pkl")
    joblib.dump(grade_le, MODEL_DIR / "grade_encoder.pkl")
    joblib.dump(prep, MODEL_DIR / "preprocessor.pkl")
    joblib.dump(prep_nn, MODEL_DIR / "nn_preprocessor.pkl")

    idim = X_tr.shape[1]
    metrics = {"risk": {}, "grade": {}, "gpa": {}}

    sklearn_models = {
        "risk": {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=14, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=5,
                random_state=RANDOM_STATE,
            ),
        },
        "grade": {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=14, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=5,
                random_state=RANDOM_STATE,
            ),
        },
        "gpa": {
            "random_forest": RandomForestRegressor(
                n_estimators=200, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=5,
                random_state=RANDOM_STATE,
            ),
        },
    }

    try:
        from xgboost import XGBClassifier, XGBRegressor
        sklearn_models["risk"]["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
        )
        sklearn_models["grade"]["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            random_state=RANDOM_STATE, eval_metric="mlogloss", verbosity=0,
        )
        sklearn_models["gpa"]["xgboost"] = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            random_state=RANDOM_STATE, verbosity=0,
        )
    except ImportError:
        print("xgboost not installed — skipping")

    y_map = {"risk": (yr_tr, yr_te), "grade": (yg_tr, yg_te), "gpa": (ygpa_tr, ygpa_te)}

    for target, models in sklearn_models.items():
        y_tr, y_te = y_map[target]
        for name, model in models.items():
            print(f"Training {target}_{name}...")
            model.fit(X_tr, y_tr)
            joblib.dump(model, MODEL_DIR / f"{target}_{name}.pkl")
            if target == "gpa":
                pred = model.predict(X_te)
                metrics[target][name] = {
                    "r2": round(float(r2_score(y_te, pred)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
                }
            else:
                pred = model.predict(X_te)
                metrics[target][name] = {
                    "accuracy": round(float(accuracy_score(y_te, pred)), 4),
                    "f1": round(float(f1_score(y_te, pred, average="weighted")), 4),
                    "precision": round(float(accuracy_score(y_te, pred)), 4),
                    "recall": round(float(accuracy_score(y_te, pred)), 4),
                }
            print(f"  -> {metrics[target][name]}")

    feature_meta = {
        "numeric_cols": ALL_NUMERIC,
        "categorical_cols": CATEGORICAL_COLS,
        "all_cols": ALL_NUMERIC + CATEGORICAL_COLS,
        "input_dim": int(idim),
        "num_grade_classes": len(grade_le.classes_),
        "risk_classes": risk_le.classes_.tolist(),
        "grade_classes": grade_le.classes_.tolist(),
    }

    with open(MODEL_DIR / "feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2)
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining complete. Models saved to:", MODEL_DIR)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
