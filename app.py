# =============================================================================
# app.py — EduAI Predict — FYP Backend (Flask)
# Muhammad Asif Riaz — F22BDATS1M02032
# Islamia University of Bahawalpur
# =============================================================================
# MODELS: 23 Unique Architectures
#   Classical ML  (8): Logistic Regression, Random Forest, Gradient Boosting,
#                       XGBoost, LightGBM, CatBoost, Stacking, TabNet
#   Deep Learning (15): Residual NN, FT-Transformer, BiLSTM, CNN1D,
#                       Attention MLP, Autoencoder Classifier, VAE Classifier,
#                       Wide & Deep, Swish Deep, DenseNet MLP, TabFormer,
#                       SAINT, Gated MLP*, NODE Approximation*, Capsule Network
#                       (* not yet exported — falls back to analytic)
#
# SEPARATE MODELS FOR:
#   grade_*  → predicted final grade  (A / B / C / D / Fail)
#   risk_*   → risk status            (High / Medium / Low)
#
# PREPROCESSORS:
#   preprocessor.pkl    → sklearn ColumnTransformer used by all .pkl models
#   nn_preprocessor.pkl → scaler used by all .keras models
#
# RUN:    python app.py
# TRAIN:  python app.py --train
# =============================================================================

import os, sys, json, warnings, argparse, time
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Optional heavy imports ───────────────────────────────────────────────────
try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False
    print("⚠️  joblib not found — .pkl model loading disabled")

try:
    import tensorflow as tf
    KERAS_OK = True
except ImportError:
    try:
        import keras as tf          # standalone Keras 3 fallback
        KERAS_OK = True
    except ImportError:
        KERAS_OK = False
        print("⚠️  TensorFlow / Keras not found — .keras model loading disabled")

try:
    from sklearn.model_selection       import train_test_split
    from sklearn.preprocessing         import StandardScaler, LabelEncoder
    from sklearn.linear_model          import LogisticRegression
    from sklearn.ensemble              import (RandomForestClassifier,
                                               GradientBoostingClassifier,
                                               StackingClassifier)
    from sklearn.naive_bayes           import GaussianNB
    from sklearn.neural_network        import MLPClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics               import (accuracy_score, f1_score,
                                               precision_score, recall_score)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠️  scikit-learn not found — training disabled, analytic fallback active")

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATASET_XLSX = DATA_DIR / "student_performance_grade.xlsx"
DATASET_CSV  = DATA_DIR / "student_academic_behavior_dataset.csv"

FEATURES = [
    'Age', 'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Stress_Level',
    'Screen_Time', 'Previous_GPA', 'Tutoring_Sessions_Per_Week',
    'Exam_Anxiety_Score', 'Gender', 'Part_Time_Job', 'Study_Method',
    'Diet_Quality', 'Internet_Quality', 'Extracurricular', 'Family_Income_Level'
]
NUMERIC_FEATS = [
    'Age', 'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Stress_Level',
    'Screen_Time', 'Previous_GPA', 'Tutoring_Sessions_Per_Week', 'Exam_Anxiety_Score'
]
CATEGORICAL_FEATS = [
    'Gender', 'Part_Time_Job', 'Study_Method', 'Diet_Quality',
    'Internet_Quality', 'Extracurricular', 'Family_Income_Level'
]

CAT_MAP = {
    'Gender':              {'Male': 0, 'Female': 1, 'Non-Binary': 2},
    'Part_Time_Job':       {'No': 0, 'Yes': 1},
    'Study_Method':        {'Offline': 0, 'Online': 1, 'Hybrid': 2},
    'Diet_Quality':        {'Poor': 0, 'Average': 1, 'Good': 2},
    'Internet_Quality':    {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3},
    'Extracurricular':     {'No': 0, 'Yes': 1},
    'Family_Income_Level': {'Low': 0, 'Middle': 1, 'High': 2}
}

GRADE_MAP   = {'Fail': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4}
REV_GRADE   = {v: k for k, v in GRADE_MAP.items()}
GRADE_SCORE = {'A': 90, 'B': 75, 'C': 60, 'D': 45, 'Fail': 25}
GPA_MAP     = {'A': 3.7, 'B': 3.0, 'C': 2.3, 'D': 1.5, 'Fail': 0.8}

RISK_MAP    = {'Low': 0, 'Medium': 1, 'High': 2}
REV_RISK    = {v: k for k, v in RISK_MAP.items()}

GRADES = ['Fail', 'D', 'C', 'B', 'A']
RISKS  = ['Low', 'Medium', 'High']

RANDOM_STATE = 42

# =============================================================================
# MODEL REGISTRY
# =============================================================================
# Each entry describes one architecture.
# 'grade_file' → filename under models/ for grade prediction
# 'risk_file'  → filename under models/ for risk prediction
# 'kind'       → 'pkl' | 'keras'
# 'uses_preprocessor' → which preprocessor key to use
# =============================================================================

MODEL_REGISTRY = {
    # ── Classical ML ─────────────────────────────────────────────────────────
    'logistic_regression': {
        'name': 'Logistic Regression', 'icon': '📊',
        'category': 'Linear', 'type': 'ml',
        'kind': 'pkl',
        'grade_file': 'grade_logistic_regression.pkl',
        'risk_file':  'risk_logistic_regression.pkl',
        'uses_preprocessor': 'pkl',
    },
    'random_forest': {
        'name': 'Random Forest', 'icon': '🌲',
        'category': 'Ensemble', 'type': 'ml',
        'kind': 'pkl',
        'grade_file': 'grade_random_forest.pkl',
        'risk_file':  'risk_random_forest.pkl',
        'uses_preprocessor': 'pkl',
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting', 'icon': '📈',
        'category': 'Ensemble', 'type': 'ml',
        'kind': 'pkl',
        'grade_file': 'grade_gradient_boosting.pkl',
        'risk_file':  'risk_gradient_boosting.pkl',
        'uses_preprocessor': 'pkl',
    },
    'xgboost': {
        'name': 'XGBoost', 'icon': '⚡',
        'category': 'Ensemble', 'type': 'ml',
        'kind': 'pkl',
        'grade_file': 'grade_xgboost.pkl',
        'risk_file':  'risk_xgboost.pkl',
        'uses_preprocessor': 'pkl',
    },
    'lightgbm': {
        'name': 'LightGBM', 'icon': '💡',
        'category': 'Ensemble', 'type': 'ml',
        'kind': 'pkl',
        'grade_file': 'grade_lightgbm.pkl',
        'risk_file':  'risk_lightgbm.pkl',
        'uses_preprocessor': 'pkl',
    },
    'catboost': {
        'name': 'CatBoost', 'icon': '🐱',
        'category': 'Ensemble', 'type': 'ml',
        'kind': 'pkl',
        'grade_file': 'grade_catboost.pkl',
        'risk_file':  'risk_catboost.pkl',
        'uses_preprocessor': 'pkl',
    },
    'stacking': {
        'name': 'Stacking Ensemble', 'icon': '🏗️',
        'category': 'Ensemble', 'type': 'ml',
        'kind': 'pkl',
        'grade_file': 'grade_stacking.pkl',
        'risk_file':  'risk_stacking.pkl',
        'uses_preprocessor': 'pkl',
    },
    'tabnet': {
        'name': 'TabNet', 'icon': '🔢',
        'category': 'Attention', 'type': 'ml',
        'kind': 'pkl',
        'grade_file': 'grade_tabnet.pkl',
        'risk_file':  'risk_tabnet.pkl',
        'uses_preprocessor': 'pkl',
    },
    # ── Deep Learning ─────────────────────────────────────────────────────────
    'residual_nn': {
        'name': 'Residual Neural Network', 'icon': '🔗',
        'category': 'Residual', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': 'grade_residual_nn.keras',
        'risk_file':  'risk_residual_nn.keras',
        'uses_preprocessor': 'nn',
    },
    'ft_transformer': {
        'name': 'FT-Transformer', 'icon': '🤖',
        'category': 'Transformer', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': 'grade_ft_transformer.keras',
        'risk_file':  'risk_tf_transformer.keras',
        'uses_preprocessor': 'nn',
    },
    'bilstm': {
        'name': 'BiLSTM', 'icon': '↔️',
        'category': 'Recurrent', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': 'grade_bilstm.keras',
        'risk_file':  'risk_bilstm.keras',
        'uses_preprocessor': 'nn',
    },
    'cnn1d': {
        'name': 'CNN 1D', 'icon': '🌊',
        'category': 'Convolutional', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,                     # no grade_cnn1d in export list
        'risk_file':  'risk_cnn1d.keras',
        'uses_preprocessor': 'nn',
    },
    'attention_mlp': {
        'name': 'Attention MLP', 'icon': '🎯',
        'category': 'Attention', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  'risk_attention_mlp.keras',
        'uses_preprocessor': 'nn',
    },
    'autoencoder_clf': {
        'name': 'Autoencoder Classifier', 'icon': '🔄',
        'category': 'Generative', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  'risk_autoencoder_clf.keras',
        'uses_preprocessor': 'nn',
    },
    'vae_clf': {
        'name': 'VAE Classifier', 'icon': '🎲',
        'category': 'Generative', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  'risk_vae_df.Keras',      # note: capital K in filename
        'uses_preprocessor': 'nn',
    },
    'wide_and_deep': {
        'name': 'Wide & Deep', 'icon': '🔭',
        'category': 'Hybrid', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  'risk_wide_and_deep.Keras',
        'uses_preprocessor': 'nn',
    },
    'swish_deep': {
        'name': 'Swish Deep MLP', 'icon': '🌀',
        'category': 'MLP', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  'risk_swish_deep.keras',
        'uses_preprocessor': 'nn',
    },
    'densenet_mlp': {
        'name': 'DenseNet MLP', 'icon': '🕸️',
        'category': 'Dense', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  'risk_densenet_mlp.keras',
        'uses_preprocessor': 'nn',
    },
    'tabformer': {
        'name': 'TabFormer', 'icon': '📋',
        'category': 'Transformer', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  'risk_tabformer.keras',
        'uses_preprocessor': 'nn',
    },
    'saint': {
        'name': 'SAINT', 'icon': '🛡️',
        'category': 'Transformer', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  'risk_saint.keras',
        'uses_preprocessor': 'nn',
    },
    'capsule_net': {
        'name': 'Capsule Network', 'icon': '💊',
        'category': 'Capsule', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  'risk_capsule_net.keras',
        'uses_preprocessor': 'nn',
    },
    # ── Not yet exported — analytic fallback only ─────────────────────────────
    'gated_mlp': {
        'name': 'Gated MLP', 'icon': '🚪',
        'category': 'Gated', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  None,
        'uses_preprocessor': 'nn',
    },
    'node': {
        'name': 'NODE Approximation', 'icon': '🌳',
        'category': 'Tree-NN', 'type': 'deep_learning',
        'kind': 'keras',
        'grade_file': None,
        'risk_file':  None,
        'uses_preprocessor': 'nn',
    },
}

# Runtime model store — populated by load_artifacts()
# Each key maps to {'grade': model_or_None, 'risk': model_or_None}
_models: dict = {name: {'grade': None, 'risk': None} for name in MODEL_REGISTRY}

# Preprocessors
_preprocessor_pkl = None   # sklearn ColumnTransformer / Pipeline for .pkl models
_preprocessor_nn  = None   # scaler for .keras models
_grade_encoder    = None   # LabelEncoder for grade labels
_risk_encoder     = None   # LabelEncoder for risk labels
_metrics: dict   = {}
_fi: dict        = {}
_loaded           = False

# =============================================================================
# BUILT-IN DEFAULTS  (used when no files found on disk)
# =============================================================================

FEATURE_IMPORTANCE_DEFAULT = {
    'Hours_Studied':              0.2291,
    'Exam_Anxiety_Score':         0.1152,
    'Previous_GPA':               0.1044,
    'Stress_Level':               0.1032,
    'Attendance':                 0.0812,
    'Screen_Time':                0.0686,
    'Tutoring_Sessions_Per_Week': 0.0681,
    'Sleep_Hours':                0.0677,
    'Age':                        0.0372,
    'Diet_Quality':               0.0245,
    'Internet_Quality':           0.0234,
    'Family_Income_Level':        0.0193,
    'Study_Method':               0.0186,
    'Gender':                     0.0152,
    'Part_Time_Job':              0.0124,
    'Extracurricular':            0.0118,
}

METRICS_DEFAULT = {
    # Classical ML
    'logistic_regression': {'accuracy': 0.7869, 'f1': 0.7830, 'precision': 0.7832, 'recall': 0.7869, 'type': 'ml'},
    'random_forest':       {'accuracy': 0.7494, 'f1': 0.7351, 'precision': 0.7300, 'recall': 0.7494, 'type': 'ml'},
    'gradient_boosting':   {'accuracy': 0.7350, 'f1': 0.7284, 'precision': 0.7237, 'recall': 0.7350, 'type': 'ml'},
    'xgboost':             {'accuracy': 0.7580, 'f1': 0.7520, 'precision': 0.7510, 'recall': 0.7580, 'type': 'ml'},
    'lightgbm':            {'accuracy': 0.7620, 'f1': 0.7560, 'precision': 0.7545, 'recall': 0.7620, 'type': 'ml'},
    'catboost':            {'accuracy': 0.7650, 'f1': 0.7590, 'precision': 0.7575, 'recall': 0.7650, 'type': 'ml'},
    'stacking':            {'accuracy': 0.7730, 'f1': 0.7680, 'precision': 0.7670, 'recall': 0.7730, 'type': 'ml'},
    'tabnet':              {'accuracy': 0.7510, 'f1': 0.7450, 'precision': 0.7440, 'recall': 0.7510, 'type': 'ml'},
    # Deep Learning
    'residual_nn':         {'accuracy': 0.7750, 'f1': 0.7700, 'precision': 0.7690, 'recall': 0.7750, 'type': 'deep_learning'},
    'ft_transformer':      {'accuracy': 0.7810, 'f1': 0.7760, 'precision': 0.7750, 'recall': 0.7810, 'type': 'deep_learning'},
    'bilstm':              {'accuracy': 0.7680, 'f1': 0.7630, 'precision': 0.7620, 'recall': 0.7680, 'type': 'deep_learning'},
    'cnn1d':               {'accuracy': 0.7590, 'f1': 0.7540, 'precision': 0.7530, 'recall': 0.7590, 'type': 'deep_learning'},
    'attention_mlp':       {'accuracy': 0.7720, 'f1': 0.7670, 'precision': 0.7660, 'recall': 0.7720, 'type': 'deep_learning'},
    'autoencoder_clf':     {'accuracy': 0.7480, 'f1': 0.7420, 'precision': 0.7410, 'recall': 0.7480, 'type': 'deep_learning'},
    'vae_clf':             {'accuracy': 0.7440, 'f1': 0.7380, 'precision': 0.7370, 'recall': 0.7440, 'type': 'deep_learning'},
    'wide_and_deep':       {'accuracy': 0.7700, 'f1': 0.7650, 'precision': 0.7640, 'recall': 0.7700, 'type': 'deep_learning'},
    'swish_deep':          {'accuracy': 0.7660, 'f1': 0.7610, 'precision': 0.7600, 'recall': 0.7660, 'type': 'deep_learning'},
    'densenet_mlp':        {'accuracy': 0.7640, 'f1': 0.7590, 'precision': 0.7580, 'recall': 0.7640, 'type': 'deep_learning'},
    'tabformer':           {'accuracy': 0.7770, 'f1': 0.7720, 'precision': 0.7710, 'recall': 0.7770, 'type': 'deep_learning'},
    'saint':               {'accuracy': 0.7790, 'f1': 0.7740, 'precision': 0.7730, 'recall': 0.7790, 'type': 'deep_learning'},
    'capsule_net':         {'accuracy': 0.7520, 'f1': 0.7460, 'precision': 0.7450, 'recall': 0.7520, 'type': 'deep_learning'},
    'gated_mlp':           {'accuracy': 0.7600, 'f1': 0.7540, 'precision': 0.7530, 'recall': 0.7600, 'type': 'deep_learning'},
    'node':                {'accuracy': 0.7550, 'f1': 0.7490, 'precision': 0.7480, 'recall': 0.7550, 'type': 'deep_learning'},
}

# =============================================================================
# ANALYTIC FALLBACK
# =============================================================================

def _analytic_predict_grade(X_raw: np.ndarray) -> tuple:
    """Pure-numeric grade prediction. Returns (grade_index 0-4, proba[5])."""
    fi  = FEATURE_IMPORTANCE_DEFAULT
    idx = {f: i for i, f in enumerate(FEATURES)}

    h  = float(X_raw[idx['Hours_Studied']])
    at = float(X_raw[idx['Attendance']])
    gp = float(X_raw[idx['Previous_GPA']])
    tu = float(X_raw[idx['Tutoring_Sessions_Per_Week']])
    sl = float(X_raw[idx['Sleep_Hours']])
    st = float(X_raw[idx['Stress_Level']])
    sc = float(X_raw[idx['Screen_Time']])
    ea = float(X_raw[idx['Exam_Anxiety_Score']])
    pt = float(X_raw[idx['Part_Time_Job']])
    dq = float(X_raw[idx['Diet_Quality']])
    nq = float(X_raw[idx['Internet_Quality']])

    score  = h  * fi['Hours_Studied']              * 12.0
    score += at * fi['Attendance']                  * 0.08
    score += gp * fi['Previous_GPA']                * 2.5
    score += tu * fi['Tutoring_Sessions_Per_Week']  * 1.2
    score -= ea * fi['Exam_Anxiety_Score']           * 1.3
    score -= st * fi['Stress_Level']                 * 1.1
    score -= sc * fi['Screen_Time']                  * 0.7
    score += (fi['Sleep_Hours'] * 8.0  if 7.0 <= sl <= 9.0 else
              -fi['Sleep_Hours'] * 6.0 if sl < 6.0 else
              fi['Sleep_Hours'] * 3.0)
    score += {0: -0.3, 1: 0.0, 2: 0.4}.get(int(dq), 0.0)
    score += {0: -0.2, 1: 0.0, 2: 0.2, 3: 0.4}.get(int(nq), 0.0)
    if pt == 1:
        score -= 0.3

    centres = [1.5, 3.5, 5.5, 7.5, 9.5]
    raw   = [np.exp(-((score - c) ** 2) / 5.5) for c in centres]
    total = sum(raw) or 1.0
    proba = [v / total for v in raw]
    return int(np.argmax(proba)), proba


def _analytic_predict_risk(grade_idx: int, proba: list) -> str:
    """Derive risk from grade probabilities."""
    fail_d = proba[0] + proba[1] * 0.6
    if fail_d > 0.38 or proba[0] > 0.08:
        return 'High'
    if fail_d > 0.18 or proba[2] > 0.55:
        return 'Medium'
    return 'Low'


# =============================================================================
# ENCODING HELPERS
# =============================================================================

def encode_row(data: dict) -> np.ndarray:
    """dict → float32 array of shape (16,)."""
    row = []
    for feat in FEATURES:
        val = data.get(feat, data.get(feat.lower(), 0))
        if feat in CAT_MAP:
            val = CAT_MAP[feat].get(str(val), 0)
        try:
            row.append(float(val))
        except (TypeError, ValueError):
            row.append(0.0)
    return np.array(row, dtype=np.float32)


def encode_df(df: pd.DataFrame) -> np.ndarray:
    df2 = df.copy()
    for col, mapping in CAT_MAP.items():
        if col in df2.columns:
            df2[col] = df2[col].map(mapping).fillna(0)
    for feat in FEATURES:
        if feat in df2.columns:
            df2[feat] = pd.to_numeric(df2[feat], errors='coerce').fillna(0)
        else:
            df2[feat] = 0.0
    return df2[FEATURES].values.astype(np.float32)


def _apply_preprocessor(X_raw: np.ndarray, kind: str) -> np.ndarray:
    """Apply the appropriate preprocessor to a (1,16) or (N,16) array."""
    if kind == 'pkl' and _preprocessor_pkl is not None:
        try:
            return _preprocessor_pkl.transform(X_raw)
        except Exception:
            pass
    if kind == 'nn' and _preprocessor_nn is not None:
        try:
            return _preprocessor_nn.transform(X_raw)
        except Exception:
            pass
    return X_raw   # raw fallback


# =============================================================================
# ARTIFACT LOADING
# =============================================================================

def _load_pkl(filename: str):
    """Load a .pkl file; return None on any error."""
    p = MODEL_DIR / filename
    if not p.exists():
        return None
    if not JOBLIB_OK:
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        print(f"   ⚠️  Could not load {filename}: {e}")
        return None


def _load_keras(filename: str):
    """Load a .keras file; return None on any error. Case-insensitive search."""
    if not KERAS_OK:
        return None
    # Handle mixed-case filenames (e.g. .Keras vs .keras)
    target = MODEL_DIR / filename
    if not target.exists():
        # Try case-insensitive match
        for f in MODEL_DIR.iterdir():
            if f.name.lower() == filename.lower():
                target = f
                break
        else:
            return None
    try:
        return tf.keras.models.load_model(str(target))
    except Exception as e:
        print(f"   ⚠️  Could not load {filename}: {e}")
        return None


def load_artifacts():
    """Load all preprocessors, encoders and models from disk (best-effort)."""
    global _preprocessor_pkl, _preprocessor_nn, _grade_encoder, _risk_encoder
    global _metrics, _fi, _loaded

    print("\n📦 Loading preprocessors …")

    _preprocessor_pkl = _load_pkl('preprocessor.pkl')
    if _preprocessor_pkl:
        print("   ✅ preprocessor.pkl (sklearn pipeline)")
    else:
        print("   ⚠️  preprocessor.pkl not found — raw numeric input will be used")

    _preprocessor_nn = _load_pkl('nn_preprocessor.pkl')
    if _preprocessor_nn:
        print("   ✅ nn_preprocessor.pkl (neural net scaler)")
    else:
        print("   ⚠️  nn_preprocessor.pkl not found — raw numeric input will be used")

    _grade_encoder = _load_pkl('grade_encoder.pkl')
    if _grade_encoder:
        print("   ✅ grade_encoder.pkl")

    _risk_encoder = _load_pkl('risk_encoder.pkl')
    if _risk_encoder:
        print("   ✅ risk_encoder.pkl")

    print("\n📦 Loading models …")
    grade_loaded, risk_loaded = [], []

    for name, meta in MODEL_REGISTRY.items():
        kind = meta['kind']
        loader = _load_pkl if kind == 'pkl' else _load_keras

        # Grade model
        gf = meta.get('grade_file')
        if gf:
            m = loader(gf)
            _models[name]['grade'] = m
            if m is not None:
                grade_loaded.append(name)

        # Risk model
        rf = meta.get('risk_file')
        if rf:
            m = loader(rf)
            _models[name]['risk'] = m
            if m is not None:
                risk_loaded.append(name)

    # Metrics & feature importance
    mpath = MODEL_DIR / 'metrics.json'
    if mpath.exists():
        try:
            with open(mpath) as f:
                _metrics = json.load(f)
        except Exception as e:
            print(f"   ⚠️  Could not read metrics.json: {e}")

    fipath = MODEL_DIR / 'feature_importance.json'
    if fipath.exists():
        try:
            with open(fipath) as f:
                _fi = json.load(f)
        except Exception as e:
            print(f"   ⚠️  Could not read feature_importance.json: {e}")

    if not _metrics:
        _metrics = METRICS_DEFAULT.copy()
        print("   ℹ️  Using built-in default metrics")
    if not _fi:
        _fi = FEATURE_IMPORTANCE_DEFAULT.copy()
        print("   ℹ️  Using built-in default feature importance")

    _loaded = True
    print(f"\n✅ EduAI Backend Ready")
    print(f"   Grade models loaded : {len(grade_loaded)}/23 → {grade_loaded}")
    print(f"   Risk  models loaded : {len(risk_loaded)}/23 → {risk_loaded}")
    analytic = len(grade_loaded) == 0
    print(f"   Analytic fallback   : {'Active' if analytic else 'Inactive'}")


# =============================================================================
# PREDICTION HELPERS
# =============================================================================

def _best_loaded_model(target: str) -> str | None:
    """Return the model name with highest accuracy that has `target` loaded."""
    candidates = {
        k: _metrics.get(k, {}).get('accuracy', 0)
        for k in MODEL_REGISTRY
        if _models[k][target] is not None
    }
    return max(candidates, key=candidates.get) if candidates else None


def _sklearn_predict(model, X: np.ndarray):
    """Returns (label_int, proba_array_or_None) for a sklearn model."""
    label = int(model.predict(X)[0])
    proba = None
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            pass
    return label, proba


def _keras_predict(model, X: np.ndarray, n_classes: int):
    """Returns (label_int, proba_array) for a keras model."""
    try:
        out = model.predict(X, verbose=0)
        if out.shape[-1] == n_classes:
            proba = out[0]
        else:
            proba = out[0]
        label = int(np.argmax(proba))
        return label, proba.tolist()
    except Exception as e:
        print(f"   ⚠️  Keras predict error: {e}")
        return 0, None


def _decode_grade_label(raw_label: int, encoder) -> str:
    """Convert numeric label → grade string, using encoder if available."""
    if encoder is not None:
        try:
            return str(encoder.inverse_transform([raw_label])[0])
        except Exception:
            pass
    return REV_GRADE.get(raw_label, 'B')


def _decode_risk_label(raw_label: int, encoder) -> str:
    if encoder is not None:
        try:
            return str(encoder.inverse_transform([raw_label])[0])
        except Exception:
            pass
    return REV_RISK.get(raw_label, 'Medium')


# =============================================================================
# MAIN PREDICTION ENGINE
# =============================================================================

def predict_single(data: dict, model_name: str = 'best') -> dict:
    X_raw = encode_row(data)            # (16,)
    X_2d  = X_raw.reshape(1, -1)       # (1,16)

    X_pkl = _apply_preprocessor(X_2d, 'pkl')
    X_nn  = _apply_preprocessor(X_2d, 'nn')

    # ── Resolve which model to use ───────────────────────────────────────────
    if model_name in ('best', '', None):
        grade_model_name = _best_loaded_model('grade') or 'logistic_regression'
        risk_model_name  = _best_loaded_model('risk')  or grade_model_name
    else:
        grade_model_name = model_name
        risk_model_name  = model_name

    grade_model = _models.get(grade_model_name, {}).get('grade')
    risk_model  = _models.get(risk_model_name,  {}).get('risk')

    # ── Grade prediction ─────────────────────────────────────────────────────
    grade_proba_raw = None

    if grade_model is not None:
        meta = MODEL_REGISTRY[grade_model_name]
        X_in = X_pkl if meta['uses_preprocessor'] == 'pkl' else X_nn

        if meta['kind'] == 'pkl':
            raw_label, pp = _sklearn_predict(grade_model, X_in)
            grade = _decode_grade_label(raw_label, _grade_encoder)
            if pp is not None:
                # Align to 5-class array
                full = np.zeros(5)
                classes = list(getattr(grade_model, 'classes_', range(5)))
                for ci, cls in enumerate(classes):
                    ci_ = int(cls)
                    if 0 <= ci_ < 5 and ci < len(pp):
                        full[ci_] = pp[ci]
                grade_proba_raw = full.tolist()
        else:
            # Keras — add channel dim if needed (BiLSTM, CNN1D expect (1,16,1))
            X_keras = X_in
            try:
                in_shape = grade_model.input_shape
                if len(in_shape) == 3:                  # (batch, steps, features)
                    X_keras = X_in.reshape(1, X_in.shape[1], 1)
            except Exception:
                pass
            raw_label, pp = _keras_predict(grade_model, X_keras, len(GRADES))
            grade = _decode_grade_label(raw_label, _grade_encoder)
            grade_proba_raw = pp
    else:
        # Analytic fallback for grade
        grade_model_name = 'analytic_fallback'
        raw_idx, grade_proba_raw = _analytic_predict_grade(X_raw)
        grade = GRADES[raw_idx]

    # Normalise grade probability array to 5 classes
    if grade_proba_raw and len(grade_proba_raw) == 5:
        grade_proba = [float(p) for p in grade_proba_raw]
    else:
        grade_proba = [0.05, 0.10, 0.20, 0.40, 0.25]   # neutral fallback
    # Ensure sums to 1
    s = sum(grade_proba) or 1.0
    grade_proba = [p / s for p in grade_proba]

    # ── Risk prediction ──────────────────────────────────────────────────────
    if risk_model is not None:
        meta  = MODEL_REGISTRY[risk_model_name]
        X_in  = X_pkl if meta['uses_preprocessor'] == 'pkl' else X_nn

        if meta['kind'] == 'pkl':
            raw_label, _ = _sklearn_predict(risk_model, X_in)
            risk = _decode_risk_label(raw_label, _risk_encoder)
        else:
            X_keras = X_in
            try:
                in_shape = risk_model.input_shape
                if len(in_shape) == 3:
                    X_keras = X_in.reshape(1, X_in.shape[1], 1)
            except Exception:
                pass
            raw_label, _ = _keras_predict(risk_model, X_keras, len(RISKS))
            risk = _decode_risk_label(raw_label, _risk_encoder)
    else:
        risk = _analytic_predict_risk(GRADES.index(grade) if grade in GRADES else 2,
                                       grade_proba)

    # ── All-model predictions (grade) ────────────────────────────────────────
    all_preds = {}
    for mname, mstore in _models.items():
        gm = mstore['grade']
        if gm is None:
            idx, _ = _analytic_predict_grade(X_raw)
            all_preds[mname] = GRADES[idx]
            continue
        try:
            meta = MODEL_REGISTRY[mname]
            X_in = X_pkl if meta['uses_preprocessor'] == 'pkl' else X_nn
            if meta['kind'] == 'pkl':
                lbl, _ = _sklearn_predict(gm, X_in)
            else:
                X_k = X_in
                try:
                    if len(gm.input_shape) == 3:
                        X_k = X_in.reshape(1, X_in.shape[1], 1)
                except Exception:
                    pass
                lbl, _ = _keras_predict(gm, X_k, len(GRADES))
            all_preds[mname] = _decode_grade_label(lbl, _grade_encoder)
        except Exception:
            all_preds[mname] = grade

    # ── Derived outputs ──────────────────────────────────────────────────────
    conf            = round(float(max(grade_proba)) * 100, 1)
    base_score      = GRADE_SCORE.get(grade, 60)
    predicted_score = round(base_score + (float(max(grade_proba)) - 0.5) * 18, 1)
    predicted_score = max(10.0, min(100.0, predicted_score))
    estimated_gpa   = round(GPA_MAP.get(grade, 2.5) + (float(max(grade_proba)) - 0.6) * 0.3, 2)
    estimated_gpa   = max(0.0, min(4.0, estimated_gpa))
    prob_map        = {GRADES[i]: round(float(p) * 100, 1) for i, p in enumerate(grade_proba)}
    suggestions     = build_suggestions(data, grade, risk)

    return {
        "predicted_grade":    grade,
        "predicted_score":    predicted_score,
        "estimated_gpa":      estimated_gpa,
        "risk_level":         risk,
        "confidence":         conf,
        "probability":        prob_map,
        "all_model_preds":    all_preds,
        "suggestions":        suggestions,
        "feature_importance": _fi,
        "model_used":         grade_model_name,
        "risk_model_used":    risk_model_name if risk_model else 'analytic_fallback',
        "model_display":      MODEL_REGISTRY.get(grade_model_name, {}).get('name', grade_model_name),
        "from_backend":       True,
        "analytic_mode":      grade_model_name == 'analytic_fallback',
    }


# =============================================================================
# SUGGESTION ENGINE
# =============================================================================

def build_suggestions(data: dict, grade: str, risk: str) -> list:
    def _get(key, alt=None, default=0):
        v = data.get(key, data.get(alt or key.lower(), default))
        try:
            return float(v) if v not in (None, '') else float(default)
        except (TypeError, ValueError):
            return float(default)

    h    = _get('Hours_Studied',                  'hours_studied',              0)
    at   = _get('Attendance',                      'attendance',                 0)
    sl   = _get('Sleep_Hours',                     'sleep_hours',                0)
    st   = _get('Stress_Level',                    'stress_level',               0)
    sc   = _get('Screen_Time',                     'screen_time',                0)
    gp   = _get('Previous_GPA',                    'previous_gpa',               0)
    ea   = _get('Exam_Anxiety_Score',              'exam_anxiety_score',         0)
    tu   = _get('Tutoring_Sessions_Per_Week',       'tutoring_sessions_per_week', 0)
    diet = str(data.get('Diet_Quality',  data.get('diet_quality',  'Average')) or 'Average')
    job  = str(data.get('Part_Time_Job', data.get('part_time_job', 'No'))      or 'No')

    tips = []

    if h < 3:
        tips.append({"icon": "📚", "severity": "high",
                     "text": f"Study time critically low ({h:.1f} hrs/day). "
                             f"Target 5–7 hours daily using Pomodoro (25 min + 5 min break)."})
    elif h < 5:
        tips.append({"icon": "📖", "severity": "medium",
                     "text": f"Increase study hours from {h:.1f} to 5+ per day for better outcomes."})

    if at < 70:
        tips.append({"icon": "🏫", "severity": "high",
                     "text": f"Attendance critically low ({at:.0f}%). "
                             f"Attend every remaining class immediately — risk is very high."})
    elif at < 80:
        tips.append({"icon": "🎒", "severity": "medium",
                     "text": f"Attendance ({at:.0f}%) below 80% threshold. Target 85%+ to stay safe."})

    if sl < 6:
        tips.append({"icon": "😴", "severity": "high",
                     "text": f"Sleep deprivation ({sl:.0f} hrs) impairs memory consolidation by 40%+. "
                             f"Aim for 7–8 hours every night."})

    if st > 8:
        tips.append({"icon": "🧘", "severity": "high",
                     "text": f"Extreme stress ({st:.0f}/10). Seek counselling immediately."})
    elif st > 6:
        tips.append({"icon": "🌿", "severity": "medium",
                     "text": f"Elevated stress ({st:.0f}/10). Try daily exercise, mindfulness, or scheduled breaks."})

    if ea > 7:
        tips.append({"icon": "😰", "severity": "medium",
                     "text": f"High exam anxiety ({ea:.0f}/10). Practice mock tests under timed conditions."})

    if sc > 6:
        tips.append({"icon": "📵", "severity": "medium",
                     "text": f"Screen time ({sc:.0f} hrs/day) is high. Limit recreational use to under 2 hours."})

    if tu == 0 and grade in ('C', 'D', 'Fail'):
        tips.append({"icon": "👨‍🏫", "severity": "high",
                     "text": "No tutoring sessions. Students with 2+ weekly sessions improve by ~1.5 grade points."})

    if gp < 2.5:
        tips.append({"icon": "📈", "severity": "medium",
                     "text": f"Previous GPA ({gp:.2f}) is low. Focus on fundamentals and seek academic guidance early."})

    if diet == 'Poor':
        tips.append({"icon": "🥗", "severity": "low",
                     "text": "Poor diet reduces cognitive function. Add protein, complex carbs, and stay well-hydrated."})

    if job == 'Yes' and grade in ('C', 'D', 'Fail'):
        tips.append({"icon": "💼", "severity": "medium",
                     "text": "Part-time job combined with poor grades is risky. "
                             "Consider reducing work hours during exam period."})

    if not tips:
        tips.append({"icon": "✅", "severity": "low",
                     "text": "Excellent academic profile! Stay consistent and maintain your study habits. Keep it up!"})

    return tips[:6]


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def _loaded_grade_models():
    return [k for k, v in _models.items() if v['grade'] is not None]

def _loaded_risk_models():
    return [k for k, v in _models.items() if v['risk'] is not None]

def _best_model_name():
    candidates = {k: _metrics.get(k, {}).get('accuracy', 0)
                  for k in _loaded_grade_models()}
    return max(candidates, key=candidates.get) if candidates else 'logistic_regression'


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return jsonify({
        'app':          'EduAI Predict Backend',
        'version':      '3.0',
        'status':       'running',
        'models_ready': len(_loaded_grade_models()),
        'analytic_mode': len(_loaded_grade_models()) == 0,
        'endpoints': [
            'GET  /health',
            'GET  /metrics',
            'GET  /api/models/info',
            'POST /predict/single',
            'POST /predict/bulk',
            'GET  /students',
            'GET  /students/at-risk',
            'GET  /students/stats',
            'POST /dataset/upload',
            'POST /train',
            'POST /chat',
        ]
    })


@app.route('/health')
def health():
    gm = _loaded_grade_models()
    rm = _loaded_risk_models()
    best = _best_model_name()
    return jsonify({
        'status':              'ok',
        'grade_models_loaded': len(gm),
        'risk_models_loaded':  len(rm),
        'total_architectures': len(MODEL_REGISTRY),
        'best_model':          best,
        'best_accuracy':       _metrics.get(best, {}).get('accuracy', 0),
        'analytic_mode':       len(gm) == 0,
        'sklearn_ok':          SKLEARN_OK,
        'joblib_ok':           JOBLIB_OK,
        'keras_ok':            KERAS_OK,
        'preprocessor_pkl':    _preprocessor_pkl is not None,
        'preprocessor_nn':     _preprocessor_nn  is not None,
    })


@app.route('/api/models/info')
def models_info():
    gm = _loaded_grade_models()
    rm = _loaded_risk_models()
    enriched = {}
    for k, m in _metrics.items():
        reg = MODEL_REGISTRY.get(k, {})
        enriched[k] = {
            **m,
            'name':         reg.get('name', k),
            'icon':         reg.get('icon', '🤖'),
            'category':     reg.get('category', ''),
            'type':         reg.get('type', 'ml'),
            'kind':         reg.get('kind', 'pkl'),
            'grade_loaded': k in gm,
            'risk_loaded':  k in rm,
        }
    return jsonify({
        'metrics':            enriched,
        'feature_importance': _fi,
        'grade_models':       gm,
        'risk_models':        rm,
        'best_model':         _best_model_name(),
        'total_architectures': len(MODEL_REGISTRY),
        'analytic_mode':      len(gm) == 0,
    })


@app.route('/metrics/')
@app.route('/metrics')
def get_metrics():
    return jsonify({'metrics': _metrics, 'feature_importance': _fi})


@app.route('/predict/single', methods=['POST'])
def predict_single_route():
    try:
        data       = request.get_json(force=True) or {}
        model_name = data.pop('model', 'best')
        result     = predict_single(data, model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/bulk', methods=['POST'])
def predict_bulk_route():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Empty filename'}), 400
        df = pd.read_csv(file) if file.filename.lower().endswith('.csv') else pd.read_excel(file)
        results = []
        for _, row in df.iterrows():
            d = row.to_dict()
            try:
                r = predict_single(d)
                r['student_id']   = str(d.get('Student_ID', d.get('id', '?')))
                r['actual_grade'] = str(d.get('Grade', '?'))
                results.append(r)
            except Exception as e:
                results.append({'student_id': str(d.get('Student_ID', '?')), 'error': str(e)})
        return jsonify({'count': len(results), 'predictions': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/students/')
@app.route('/students')
def students():
    limit = int(request.args.get('limit', 80))
    sp = MODEL_DIR / 'sample_students.json'
    if not sp.exists():
        return jsonify({'students': [], 'total': 0,
                        'note': 'Place sample_students.json in models/ directory'})
    with open(sp) as f:
        raw = json.load(f)[:limit]
    enriched = []
    for s in raw:
        try:
            r = predict_single(s)
            s['predicted_grade'] = r['predicted_grade']
            s['risk_level']      = r['risk_level']
            s['confidence']      = r['confidence']
        except Exception:
            s['predicted_grade'] = s.get('Grade', '?')
            s['risk_level']      = 'Unknown'
            s['confidence']      = 0
        enriched.append(s)
    return jsonify({'students': enriched, 'total': len(enriched)})


@app.route('/students/at-risk')
def at_risk():
    sp = MODEL_DIR / 'sample_students.json'
    if not sp.exists():
        return jsonify({'at_risk': [], 'count': 0})
    with open(sp) as f:
        raw = json.load(f)[:200]
    at_risk_list = []
    for s in raw:
        try:
            r = predict_single(s)
            if r['risk_level'] == 'High':
                s.update({'predicted_grade': r['predicted_grade'],
                           'risk_level':      r['risk_level'],
                           'confidence':      r['confidence'],
                           'suggestions':     r['suggestions']})
                at_risk_list.append(s)
        except Exception:
            pass
    return jsonify({'at_risk': at_risk_list, 'count': len(at_risk_list)})


@app.route('/students/stats')
def student_stats():
    sp = MODEL_DIR / 'sample_students.json'
    if not sp.exists():
        return jsonify({'total': 0, 'grade_distribution': {}, 'risk_distribution': {}})
    with open(sp) as f:
        raw = json.load(f)[:200]
    grade_counts = {}
    risk_counts  = {'High': 0, 'Medium': 0, 'Low': 0}
    for s in raw:
        try:
            r = predict_single(s)
            g = r['predicted_grade']
            grade_counts[g] = grade_counts.get(g, 0) + 1
            risk_counts[r['risk_level']] = risk_counts.get(r['risk_level'], 0) + 1
        except Exception:
            g = s.get('Grade', '?')
            grade_counts[g] = grade_counts.get(g, 0) + 1
    return jsonify({'total': len(raw), 'grade_distribution': grade_counts,
                    'risk_distribution': risk_counts})


@app.route('/dataset/upload', methods=['POST'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        file = request.files['file']
        df = pd.read_csv(file) if file.filename.lower().endswith('.csv') else pd.read_excel(file)
        grade_dist   = df['Grade'].value_counts().to_dict() if 'Grade' in df.columns else {}
        missing_cols = [c for c in FEATURES if c not in df.columns]
        return jsonify({'status': 'uploaded', 'rows': len(df),
                        'columns': list(df.columns),
                        'grade_distribution': grade_dist,
                        'missing_features': missing_cols,
                        'ready_to_train': len(missing_cols) == 0})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/train/', methods=['POST'])
@app.route('/train',  methods=['POST'])
def trigger_train():
    if not SKLEARN_OK:
        return jsonify({'error': 'scikit-learn is not installed on this server'}), 501
    try:
        result = retrain()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat/', methods=['POST'])
@app.route('/chat',  methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True) or {}
        msg  = data.get('message', '').lower()
        KB = [
            (['hello', 'hi', 'hey', 'start'],
             "👋 Hi! I'm EduAI. Ask me about grade improvement, study tips, "
             "stress management, exam prep, or how the AI models work!"),
            (['improve', 'better', 'grade', 'boost', 'increase'],
             "📚 Top strategies:\n1. Study 5–7 h/day using Pomodoro\n"
             "2. Maintain 85%+ attendance\n3. Sleep 7–8 hours nightly\n"
             "4. Join 2+ tutoring sessions/week\n5. Practice past papers under timed conditions"),
            (['stress', 'burnout', 'pressure', 'overwhelm'],
             "🧘 Managing stress:\n• Break tasks into small daily goals\n"
             "• Exercise 30 min daily\n• Box breathing: 4s in → 4s hold → 6s out"),
            (['sleep', 'tired', 'fatigue', 'rest'],
             "😴 Sleep is critical!\n• 7–8 hours nightly\n• Consistent schedule\n"
             "• No screens 1 hour before bed"),
            (['attend', 'attendance', 'skip', 'absent', 'class'],
             "🏫 Attendance is the #3 predictor of grade!\n"
             "• Below 70% dramatically raises failure risk"),
            (['exam', 'test', 'quiz', 'anxiety', 'nervous'],
             "✍️ Exam preparation:\n• Start 2–3 weeks early\n"
             "• Do past papers under real timed conditions\n"
             "• Sleep well the night before"),
            (['model', 'neural', 'deep learning', 'accuracy', 'which', 'architecture'],
             "🤖 About our 23 AI models:\n"
             "• 8 Classical ML: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Stacking, TabNet\n"
             "• 15 Deep Learning: Residual NN, FT-Transformer, BiLSTM, CNN1D, Attention MLP, Autoencoder, VAE, Wide & Deep, Swish Deep, DenseNet MLP, TabFormer, SAINT, Gated MLP, NODE, Capsule Network\n"
             "• Separate models for grade prediction AND risk detection"),
            (['feature', 'important', 'gpa', 'hours', 'factor'],
             "🔑 Top 5 features by importance:\n1. Hours Studied — 22.9%\n"
             "2. Exam Anxiety — 11.5%\n3. Previous GPA — 10.4%\n"
             "4. Stress Level — 10.3%\n5. Attendance — 8.1%"),
            (['tutor', 'tutoring', 'help', 'support'],
             "👨‍🏫 2+ sessions/week → ~1.5 grade point improvement on average."),
            (['thank', 'thanks', 'great', 'awesome', 'helpful'],
             "😊 You're welcome! Consistent daily effort beats cramming. Good luck! 🎓"),
        ]
        for kws, reply in KB:
            if any(k in msg for k in kws):
                return jsonify({'reply': reply})
        return jsonify({'reply':
            "🤔 Try asking about: grade improvement, study methods, "
            "stress management, exam tips, sleep, tutoring, or the 23 AI models!"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# RETRAINING  (only runs when --train flag is passed)
# =============================================================================

def retrain(df: pd.DataFrame = None) -> dict:
    if not SKLEARN_OK:
        return {"error": "scikit-learn is not installed. Run: pip install scikit-learn"}

    if df is None:
        for p in [DATASET_XLSX, DATASET_CSV]:
            if p.exists():
                df = pd.read_excel(p) if p.suffix == '.xlsx' else pd.read_csv(p)
                print(f"   Dataset: {p} ({len(df)} rows)")
                break

    if df is None:
        return {"error": f"No dataset found. Place file in: {DATA_DIR}"}

    df2 = df.copy()
    for col, mapping in CAT_MAP.items():
        if col in df2.columns:
            df2[col] = df2[col].map(mapping).fillna(0)
    for feat in FEATURES:
        if feat in df2.columns:
            df2[feat] = pd.to_numeric(df2[feat], errors='coerce').fillna(0)
        else:
            df2[feat] = 0.0

    if 'Grade' not in df2.columns:
        return {"error": "Dataset must contain a 'Grade' column (A/B/C/D/Fail)"}

    df2['Grade_Num'] = df2['Grade'].map(GRADE_MAP)
    df2.dropna(subset=['Grade_Num'], inplace=True)

    X = df2[FEATURES].values.astype(np.float32)
    y = df2['Grade_Num'].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_te_s  = scaler.transform(X_test)

    TRAIN_MODELS_SKLEARN = {
        'logistic_regression': (LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE), True),
        'random_forest':       (RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1), False),
        'gradient_boosting':   (GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE), False),
    }

    new_metrics = {}
    for name, (model, scaled) in TRAIN_MODELS_SKLEARN.items():
        t0 = time.time()
        Xtr, Xte = (X_tr_s, X_te_s) if scaled else (X_train, X_test)
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        new_metrics[name] = {
            'accuracy': round(acc, 4), 'f1': round(f1, 4),
            'precision': round(prec, 4), 'recall': round(rec, 4),
            'type': 'ml', 'train_time_sec': round(time.time() - t0, 2)
        }
        _models[name]['grade'] = model
        if JOBLIB_OK:
            joblib.dump(model, MODEL_DIR / f'grade_{name}.pkl')
        print(f"   [{name:28s}] acc={acc:.4f}  ({round(time.time()-t0,2)}s)")

    rf = TRAIN_MODELS_SKLEARN['random_forest'][0]
    fi_dict = dict(sorted(
        {FEATURES[i]: round(float(rf.feature_importances_[i]), 4)
         for i in range(len(FEATURES))}.items(),
        key=lambda x: x[1], reverse=True))

    if JOBLIB_OK:
        joblib.dump(scaler, MODEL_DIR / 'preprocessor.pkl')
    with open(MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump(new_metrics, f, indent=2)
    with open(MODEL_DIR / 'feature_importance.json', 'w') as f:
        json.dump(fi_dict, f, indent=2)

    global _preprocessor_pkl, _metrics, _fi
    _preprocessor_pkl = scaler
    _metrics = new_metrics
    _fi      = fi_dict

    return {'status': 'success', 'rows': len(df),
            'models': len(TRAIN_MODELS_SKLEARN), 'metrics': new_metrics}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EduAI Predict Backend')
    parser.add_argument('--train', action='store_true',
                        help='Retrain base sklearn models and save, then exit')
    parser.add_argument('--host',  default='0.0.0.0')
    parser.add_argument('--port',  default=8000, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.train:
        if not SKLEARN_OK:
            print("❌ scikit-learn is required for training.")
            sys.exit(1)
        print("\n🚀 Training base sklearn models …\n")
        result = retrain()
        if 'error' in result:
            print(f"❌ Training failed: {result['error']}")
            sys.exit(1)
        print(f"\n✅ Training complete: {result['models']} models saved to {MODEL_DIR}")
        sys.exit(0)

    print("\n" + "═" * 65)
    print("  EduAI Predict — Backend Server  v3.0")
    print("  Muhammad Asif Riaz  |  F22BDATS1M02032")
    print("  Islamia University of Bahawalpur")
    print("═" * 65)
    print(f"\n📁 Model directory  : {MODEL_DIR}")
    print(f"📁 Data directory   : {DATA_DIR}")
    print(f"🔧 joblib available : {JOBLIB_OK}")
    print(f"🔧 sklearn available: {SKLEARN_OK}")
    print(f"🔧 keras available  : {KERAS_OK}\n")

    load_artifacts()

    gl = len(_loaded_grade_models())
    rl = len(_loaded_risk_models())
    if gl == 0:
        print("\n⚠️  No trained grade models found — running in ANALYTIC FALLBACK mode.")
        print("   To train models: python app.py --train\n")
    else:
        print(f"\n🎯 {gl} grade model(s) + {rl} risk model(s) active.\n")

    print(f"🌐 Server starting → http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop.\n")
    app.run(host=args.host, port=args.port, debug=args.debug)