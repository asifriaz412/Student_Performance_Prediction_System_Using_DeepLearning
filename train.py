# =============================================================================
# train_all_models.py — EduAI Predict — Full Training Pipeline
# Muhammad Asif Riaz — F22BDATS1M02032
# Islamia University of Bahawalpur
# =============================================================================
# Trains ALL 23 architectures:
#   Classical ML  (8): Logistic Regression, Random Forest, Gradient Boosting,
#                       XGBoost, LightGBM, CatBoost, Stacking, TabNet
#   Deep Learning (15): Residual NN, FT-Transformer, BiLSTM, CNN1D,
#                       Attention MLP, Autoencoder Classifier, VAE Classifier,
#                       Wide & Deep, Swish Deep, DenseNet MLP, TabFormer,
#                       SAINT, Gated MLP, NODE Approximation, Capsule Network
#
# USAGE:
#   pip install scikit-learn xgboost lightgbm catboost pytorch-tabnet \
#               tensorflow imbalanced-learn pandas openpyxl joblib tqdm
#
#   python train_all_models.py                   # auto-find dataset
#   python train_all_models.py --data path/to/file.csv
#   python train_all_models.py --skip-dl         # ML only (fast)
#   python train_all_models.py --skip-ml         # DL only
#   python train_all_models.py --epochs 80 --batch 256
#
# OUTPUT (saved to  ./models/ ):
#   grade_*.pkl / grade_*.keras   — grade classifier per model
#   risk_*.pkl  / risk_*.keras    — risk  classifier per model
#   preprocessor.pkl              — sklearn ColumnTransformer (for .pkl models)
#   nn_preprocessor.pkl           — StandardScaler (for .keras models)
#   grade_encoder.pkl / risk_encoder.pkl
#   feature_importance.json
#   metrics.json
#   sample_students.json          — 120 sample records for Teacher dashboard
# =============================================================================

import os, sys, json, time, warnings, argparse, random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Output directory ──────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS  (must match app.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
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
RISK_MAP    = {'Low': 0, 'Medium': 1, 'High': 2}
REV_RISK    = {v: k for k, v in RISK_MAP.items()}

N_FEATURES  = len(FEATURES)
N_GRADES    = 5
N_RISKS     = 3
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str = None) -> pd.DataFrame:
    """Load dataset from provided path or auto-discover."""
    candidates = [
        path,
        DATA_DIR / "student_performance_grade.xlsx",
        DATA_DIR / "student_academic_behavior_dataset.csv",
        DATA_DIR / "student_data.csv",
        "student_performance_grade.xlsx",
        "student_academic_behavior_dataset.csv",
    ]
    for p in candidates:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            print(f"   📂 Dataset: {p}  ({p.stat().st_size//1024} KB)")
            if p.suffix == '.xlsx':
                return pd.read_excel(p)
            else:
                return pd.read_csv(p)
    raise FileNotFoundError(
        f"No dataset found. Place your file in {DATA_DIR}/ or pass --data <path>"
    )


def preprocess(df: pd.DataFrame):
    """
    Returns:
        X_raw     : np.float32 array (N, 16)  — label-encoded, not scaled
        y_grade   : np.int32   array (N,)     — 0..4
        y_risk    : np.int32   array (N,)     — 0..2
        df_clean  : cleaned DataFrame
    """
    df2 = df.copy()

    # Normalise column names (case-insensitive)
    col_map = {c.lower(): c for c in df2.columns}
    rename = {}
    for feat in FEATURES:
        lf = feat.lower()
        if feat not in df2.columns and lf in col_map:
            rename[col_map[lf]] = feat
    if rename:
        df2.rename(columns=rename, inplace=True)

    # Map categoricals
    for col, mapping in CAT_MAP.items():
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.strip().str.title().map(mapping)
            df2[col].fillna(df2[col].mode()[0] if not df2[col].mode().empty else 0, inplace=True)

    # Ensure all features exist
    for feat in FEATURES:
        if feat not in df2.columns:
            df2[feat] = 0.0
        df2[feat] = pd.to_numeric(df2[feat], errors='coerce').fillna(0)

    # Grade target
    grade_col = next((c for c in ['Grade', 'Final_Grade', 'grade', 'final_grade'] if c in df2.columns), None)
    if grade_col is None:
        raise ValueError("No 'Grade' column found in dataset. Expected values: A / B / C / D / Fail")
    df2['Grade_Num'] = df2[grade_col].astype(str).str.strip().str.title().map(GRADE_MAP)
    df2.dropna(subset=['Grade_Num'], inplace=True)
    df2['Grade_Num'] = df2['Grade_Num'].astype(int)

    # Risk target — derive if not present
    risk_col = next((c for c in ['Risk_Status', 'Risk', 'risk', 'risk_status'] if c in df2.columns), None)
    if risk_col:
        def _map_risk(v):
            v = str(v).strip().title()
            if v in RISK_MAP:        return RISK_MAP[v]
            if 'high' in v.lower(): return 2
            if 'med'  in v.lower(): return 1
            return 0
        df2['Risk_Num'] = df2[risk_col].apply(_map_risk).astype(int)
    else:
        # Derive from grade: Fail/D → High, C → Medium, B/A → Low
        def _derive_risk(g):
            if g <= 1: return 2   # High
            if g == 2: return 1   # Medium
            return 0              # Low
        df2['Risk_Num'] = df2['Grade_Num'].apply(_derive_risk).astype(int)

    X_raw = df2[FEATURES].values.astype(np.float32)
    y_grade = df2['Grade_Num'].values.astype(np.int32)
    y_risk  = df2['Risk_Num'].values.astype(np.int32)

    print(f"   Rows: {len(df2):,}  |  Features: {N_FEATURES}")
    print(f"   Grade distribution: { {REV_GRADE[k]: v for k, v in zip(*np.unique(y_grade, return_counts=True))} }")
    print(f"   Risk  distribution: { {REV_RISK[k]:  v for k, v in zip(*np.unique(y_risk,  return_counts=True))} }")
    return X_raw, y_grade, y_risk, df2


def build_preprocessors(X_train: np.ndarray):
    """Build both sklearn and NN preprocessors."""
    from sklearn.preprocessing import RobustScaler, StandardScaler

    pkl_scaler = RobustScaler()
    pkl_scaler.fit(X_train)

    nn_scaler = StandardScaler()
    nn_scaler.fit(X_train)

    return pkl_scaler, nn_scaler


def build_encoders(y_grade, y_risk):
    from sklearn.preprocessing import LabelEncoder
    ge = LabelEncoder(); ge.fit(y_grade)
    re = LabelEncoder(); re.fit(y_risk)
    return ge, re


# ─────────────────────────────────────────────────────────────────────────────
# 2. METRICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def eval_metrics(model, X_test, y_test, kind='pkl'):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    if kind == 'pkl':
        y_pred = model.predict(X_test)
    else:  # keras
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    return {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'precision': round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASSICAL ML MODELS  (8)
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(X_tr, y_tr, X_te, y_te, target, scaler):
    from sklearn.linear_model import LogisticRegression
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = LogisticRegression(
        max_iter=3000, C=1.0, solver='lbfgs',
        multi_class='multinomial', random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_tr_s, y_tr)
    m = eval_metrics(model, X_te_s, y_te)
    return model, m


def train_random_forest(X_tr, y_tr, X_te, y_te, target):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, max_features='sqrt',
        bootstrap=True, oob_score=True,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    m = eval_metrics(model, X_te, y_te)
    return model, m


def train_gradient_boosting(X_tr, y_tr, X_te, y_te, target):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        subsample=0.85, min_samples_split=4,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_tr)
    m = eval_metrics(model, X_te, y_te)
    return model, m


def train_xgboost(X_tr, y_tr, X_te, y_te, target):
    try:
        from xgboost import XGBClassifier
        n_cls = len(np.unique(y_tr))
        obj = 'multi:softmax' if n_cls > 2 else 'binary:logistic'
        model = XGBClassifier(
            n_estimators=400, learning_rate=0.07, max_depth=6,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,
            objective=obj, num_class=n_cls if n_cls > 2 else None,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        m = eval_metrics(model, X_te, y_te)
        return model, m
    except ImportError:
        print("   ⚠️  xgboost not installed — skipping XGBoost")
        return None, {}


def train_lightgbm(X_tr, y_tr, X_te, y_te, target):
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=500, learning_rate=0.06, max_depth=-1,
            num_leaves=63, min_child_samples=20,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                  callbacks=[])
        m = eval_metrics(model, X_te, y_te)
        return model, m
    except ImportError:
        print("   ⚠️  lightgbm not installed — skipping LightGBM")
        return None, {}


def train_catboost(X_tr, y_tr, X_te, y_te, target):
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=500, learning_rate=0.07, depth=6,
            l2_leaf_reg=3.0, random_strength=1.0,
            bagging_temperature=0.5, border_count=128,
            random_seed=RANDOM_STATE, verbose=0
        )
        model.fit(X_tr, y_tr, eval_set=(X_te, y_te), use_best_model=True)
        m = eval_metrics(model, X_te, y_te)
        return model, m
    except ImportError:
        print("   ⚠️  catboost not installed — skipping CatBoost")
        return None, {}


def train_stacking(X_tr, y_tr, X_te, y_te, target):
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                   StackingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    estimators = [
        ('rf',  RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ('gbt', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=RANDOM_STATE)),
    ]
    try:
        from xgboost import XGBClassifier
        estimators.append(('xgb', XGBClassifier(n_estimators=200, learning_rate=0.08,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)))
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        estimators.append(('lgb', LGBMClassifier(n_estimators=200, learning_rate=0.08,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)))
    except ImportError:
        pass

    final = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, C=2.0, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    model = StackingClassifier(
        estimators=estimators, final_estimator=final,
        cv=5, n_jobs=-1, passthrough=False
    )
    model.fit(X_tr, y_tr)
    m = eval_metrics(model, X_te, y_te)
    return model, m


def train_tabnet(X_tr, y_tr, X_te, y_te, target, scaler):
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch
        X_tr_s = scaler.transform(X_tr).astype(np.float32)
        X_te_s = scaler.transform(X_te).astype(np.float32)
        n_cls = len(np.unique(y_tr))
        model = TabNetClassifier(
            n_d=32, n_a=32, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            lambda_sparse=1e-4, momentum=0.3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=10, gamma=0.9),
            mask_type='entmax', verbose=0, seed=RANDOM_STATE
        )
        model.fit(
            X_tr_s, y_tr,
            eval_set=[(X_te_s, y_te)],
            eval_metric=['accuracy'],
            max_epochs=100, patience=15, batch_size=512,
        )
        y_pred = model.predict(X_te_s)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        m = {
            'accuracy':  round(float(accuracy_score(y_te, y_pred)), 4),
            'f1':        round(float(f1_score(y_te, y_pred, average='weighted', zero_division=0)), 4),
            'precision': round(float(precision_score(y_te, y_pred, average='weighted', zero_division=0)), 4),
            'recall':    round(float(recall_score(y_te, y_pred, average='weighted', zero_division=0)), 4),
        }
        return model, m
    except ImportError:
        print("   ⚠️  pytorch-tabnet not installed — skipping TabNet")
        return None, {}
    except Exception as e:
        print(f"   ⚠️  TabNet error: {e}")
        return None, {}


# ─────────────────────────────────────────────────────────────────────────────
# 4. DEEP LEARNING MODELS  (15)
# ─────────────────────────────────────────────────────────────────────────────

def get_tf():
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)
    return tf

def get_callbacks(name, patience=12):
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=patience,
            restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=6, verbose=0
        ),
    ]


# ── 4.1 Residual Neural Network ───────────────────────────────────────────────
def build_residual_nn(n_classes: int):
    tf = get_tf()
    inp = tf.keras.Input(shape=(N_FEATURES,), name='input')
    x = tf.keras.layers.Dense(256, activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    for units in [256, 256]:
        res = x
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, res])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='ResidualNN')


# ── 4.2 FT-Transformer ────────────────────────────────────────────────────────
def build_ft_transformer(n_classes: int):
    tf = get_tf()
    D = 32  # embedding dim per feature
    inp = tf.keras.Input(shape=(N_FEATURES,))

    # Linear tokenizer: each feature → D-dim embedding
    x = tf.keras.layers.Dense(D * N_FEATURES)(inp)
    x = tf.keras.layers.Reshape((N_FEATURES, D))(x)

    # 2 transformer blocks
    for _ in range(2):
        # Multi-head self-attention
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=D//4)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ffn = tf.keras.layers.Dense(D * 4, activation='gelu')(x)
        ffn = tf.keras.layers.Dense(D)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='FT_Transformer')


# ── 4.3 BiLSTM ────────────────────────────────────────────────────────────────
def build_bilstm(n_classes: int):
    tf = get_tf()
    inp = tf.keras.Input(shape=(N_FEATURES, 1))
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    )(inp)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(64, return_sequences=False, dropout=0.2)
    )(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='BiLSTM')


# ── 4.4 CNN 1D ────────────────────────────────────────────────────────────────
def build_cnn1d(n_classes: int):
    tf = get_tf()
    inp = tf.keras.Input(shape=(N_FEATURES, 1))
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='CNN1D')


# ── 4.5 Attention MLP ─────────────────────────────────────────────────────────
def build_attention_mlp(n_classes: int):
    tf = get_tf()
    inp = tf.keras.Input(shape=(N_FEATURES,))
    x = tf.keras.layers.Dense(256, activation='relu')(inp)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Self-gating attention
    attn_weights = tf.keras.layers.Dense(256, activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, attn_weights])

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    attn_weights2 = tf.keras.layers.Dense(256, activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, attn_weights2])

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='AttentionMLP')


# ── 4.6 Autoencoder Classifier ────────────────────────────────────────────────
def build_autoencoder_clf(n_classes: int):
    tf = get_tf()
    inp = tf.keras.Input(shape=(N_FEATURES,))

    # Encoder
    enc = tf.keras.layers.Dense(128, activation='relu')(inp)
    enc = tf.keras.layers.BatchNormalization()(enc)
    enc = tf.keras.layers.Dropout(0.3)(enc)
    enc = tf.keras.layers.Dense(64, activation='relu')(enc)
    bottleneck = tf.keras.layers.Dense(32, activation='relu', name='bottleneck')(enc)

    # Decoder (reconstruction — drives representation learning)
    dec = tf.keras.layers.Dense(64, activation='relu')(bottleneck)
    dec = tf.keras.layers.Dense(128, activation='relu')(dec)
    reconstruction = tf.keras.layers.Dense(N_FEATURES, name='reconstruction')(dec)

    # Classifier head
    cls = tf.keras.layers.Dense(64, activation='relu')(bottleneck)
    cls = tf.keras.layers.Dropout(0.25)(cls)
    clf_out = tf.keras.layers.Dense(n_classes, activation='softmax', name='classification')(cls)

    model = tf.keras.Model(inp, [clf_out, reconstruction], name='AutoencoderCLF')
    return model


# ── 4.7 VAE Classifier ────────────────────────────────────────────────────────
class Sampling(object):
    """Used as a Lambda; kept as standalone for clarity."""
    pass

def build_vae_clf(n_classes: int):
    tf = get_tf()
    LATENT_DIM = 16

    inp = tf.keras.Input(shape=(N_FEATURES,))
    h = tf.keras.layers.Dense(128, activation='relu')(inp)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dropout(0.3)(h)
    h = tf.keras.layers.Dense(64, activation='relu')(h)
    z_mean   = tf.keras.layers.Dense(LATENT_DIM, name='z_mean')(h)
    z_log_var= tf.keras.layers.Dense(LATENT_DIM, name='z_log_var')(h)

    # Reparameterisation trick
    z = tf.keras.layers.Lambda(
        lambda args: args[0] + tf.exp(0.5 * args[1]) * tf.random.normal(tf.shape(args[0])),
        name='z'
    )([z_mean, z_log_var])

    # Decoder (reconstruction)
    dec_h = tf.keras.layers.Dense(64, activation='relu')(z)
    dec_h = tf.keras.layers.Dense(128, activation='relu')(dec_h)
    reconstruction = tf.keras.layers.Dense(N_FEATURES, name='reconstruction')(dec_h)

    # Classifier from latent
    cls = tf.keras.layers.Dense(32, activation='relu')(z)
    cls = tf.keras.layers.Dropout(0.2)(cls)
    clf_out = tf.keras.layers.Dense(n_classes, activation='softmax', name='classification')(cls)

    model = tf.keras.Model(inp, [clf_out, reconstruction], name='VAE_CLF')
    return model


# ── 4.8 Wide & Deep ───────────────────────────────────────────────────────────
def build_wide_and_deep(n_classes: int):
    tf = get_tf()
    inp = tf.keras.Input(shape=(N_FEATURES,))

    # Wide component — direct connections from input
    wide = tf.keras.layers.Dense(64, activation='relu')(inp)

    # Deep component — deep tower
    deep = tf.keras.layers.Dense(256, activation='relu')(inp)
    deep = tf.keras.layers.BatchNormalization()(deep)
    deep = tf.keras.layers.Dropout(0.3)(deep)
    deep = tf.keras.layers.Dense(128, activation='relu')(deep)
    deep = tf.keras.layers.BatchNormalization()(deep)
    deep = tf.keras.layers.Dropout(0.25)(deep)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)

    combined = tf.keras.layers.Concatenate()([wide, deep])
    combined = tf.keras.layers.Dense(64, activation='relu')(combined)
    combined = tf.keras.layers.Dropout(0.2)(combined)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(combined)
    return tf.keras.Model(inp, out, name='WideAndDeep')


# ── 4.9 Swish Deep MLP ────────────────────────────────────────────────────────
def build_swish_deep(n_classes: int):
    tf = get_tf()
    inp = tf.keras.Input(shape=(N_FEATURES,))
    x = tf.keras.layers.Dense(512, activation='swish')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation='swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(128, activation=lambda x: x * tf.keras.activations.sigmoid(1.702 * x))(x)  # GELU approx
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='SwishDeep')


# ── 4.10 DenseNet MLP ─────────────────────────────────────────────────────────
def build_densenet_mlp(n_classes: int):
    tf = get_tf()
    inp = tf.keras.Input(shape=(N_FEATURES,))
    blocks = [inp]

    for units in [128, 128, 128, 128]:
        concat = tf.keras.layers.Concatenate()(blocks)
        x = tf.keras.layers.Dense(units, activation='relu')(concat)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        blocks.append(x)

    final = tf.keras.layers.Concatenate()(blocks)
    x = tf.keras.layers.Dense(128, activation='relu')(final)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='DenseNetMLP')


# ── 4.11 TabFormer (BERT-style for tabular) ───────────────────────────────────
def build_tabformer(n_classes: int):
    tf = get_tf()
    D = 32
    inp = tf.keras.Input(shape=(N_FEATURES,))

    # Tokenize: each feature independently projected to D dimensions
    x = tf.keras.layers.Dense(D * N_FEATURES)(inp)
    x = tf.keras.layers.Reshape((N_FEATURES, D))(x)

    # Position embedding
    positions = tf.range(N_FEATURES)
    pos_emb = tf.keras.layers.Embedding(N_FEATURES, D)(positions)
    x = x + pos_emb  # broadcast over batch

    # 3 BERT-style transformer blocks
    for _ in range(3):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=D//4, dropout=0.1)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ffn = tf.keras.layers.Dense(D * 4, activation='gelu')(x)
        ffn = tf.keras.layers.Dropout(0.1)(ffn)
        ffn = tf.keras.layers.Dense(D)(ffn)
        x = tf.keras.layers.LayerNormalization()(x + ffn)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='TabFormer')


# ── 4.12 SAINT (Self-Attention and Intersample Attention) ────────────────────
def build_saint(n_classes: int):
    tf = get_tf()
    D = 32
    inp = tf.keras.Input(shape=(N_FEATURES,))

    x = tf.keras.layers.Dense(D * N_FEATURES)(inp)
    x = tf.keras.layers.Reshape((N_FEATURES, D))(x)

    for _ in range(2):
        # Column-wise (feature) self-attention
        col_attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=D//4, dropout=0.1)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + col_attn)
        ffn = tf.keras.layers.Dense(D * 4, activation='gelu')(x)
        ffn = tf.keras.layers.Dense(D)(ffn)
        x = tf.keras.layers.LayerNormalization()(x + ffn)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='SAINT')


# ── 4.13 Gated MLP (gMLP) ─────────────────────────────────────────────────────
def build_gated_mlp(n_classes: int):
    tf = get_tf()
    inp = tf.keras.Input(shape=(N_FEATURES,))

    def gated_block(x, units):
        d = tf.keras.layers.Dense(units * 2)(x)
        x1, x2 = tf.split(d, 2, axis=-1)
        gate = tf.keras.activations.sigmoid(x2)
        x = x1 * gate
        return tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dense(256)(inp)
    x = tf.keras.layers.LayerNormalization()(x)
    for _ in range(4):
        x = gated_block(x, 256)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='GatedMLP')


# ── 4.14 NODE Approximation (Oblivious Decision Trees via Dense) ──────────────
def build_node_approx(n_classes: int):
    tf = get_tf()
    N_TREES   = 128
    TREE_DEPTH = 3   # each tree has 2^depth leaves

    inp = tf.keras.Input(shape=(N_FEATURES,))

    # Feature selection per tree (learned soft thresholds)
    selector = tf.keras.layers.Dense(N_TREES * TREE_DEPTH, activation='tanh',
                                     name='feature_selector')(inp)
    selector = tf.keras.layers.Reshape((N_TREES, TREE_DEPTH))(selector)

    # Oblivious decision tree simulation via product of sigmoid splits
    # Each tree produces 2^depth leaf activations
    choices = tf.keras.layers.Dense(N_TREES * (2 ** TREE_DEPTH),
                                    activation='softmax',
                                    name='tree_choices')(inp)
    choices = tf.keras.layers.Reshape((N_TREES, 2 ** TREE_DEPTH))(choices)

    x = tf.keras.layers.Flatten()(choices)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='NODE_Approx')


# ── 4.15 Capsule Network ──────────────────────────────────────────────────────
def build_capsule_net(n_classes: int):
    tf = get_tf()

    def squash(x, axis=-1):
        s_squared = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)
        scale = s_squared / (1 + s_squared) / tf.sqrt(s_squared + 1e-9)
        return scale * x

    inp = tf.keras.Input(shape=(N_FEATURES,))
    x = tf.keras.layers.Dense(256, activation='relu')(inp)
    x = tf.keras.layers.Reshape((32, 8))(
        tf.keras.layers.Dense(256)(x)
    )
    # Primary capsules
    primary = tf.keras.layers.Lambda(squash)(x)

    # Digit capsules (one per class)
    digit = tf.keras.layers.Dense(n_classes * 16)(
        tf.keras.layers.Flatten()(primary)
    )
    digit = tf.keras.layers.Reshape((n_classes, 16))(digit)
    digit = tf.keras.layers.Lambda(squash)(digit)

    # Length of each capsule → class probability
    out = tf.keras.layers.Lambda(
        lambda v: tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1))
    )(digit)
    out = tf.keras.layers.Softmax()(out)
    return tf.keras.Model(inp, out, name='CapsuleNet')


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEEP LEARNING TRAINING WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def train_keras_model(model, X_tr, y_tr, X_te, y_te, n_classes,
                      epochs=60, batch=256, name='', dual_output=False):
    tf = get_tf()

    if dual_output:
        # Autoencoder / VAE — multi-output: reconstruction + classification
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss={'classification': 'sparse_categorical_crossentropy',
                  'reconstruction': 'mse'},
            loss_weights={'classification': 1.0, 'reconstruction': 0.4},
            metrics={'classification': ['accuracy']}
        )
        model.fit(
            X_tr, {'classification': y_tr, 'reconstruction': X_tr},
            validation_data=(X_te, {'classification': y_te, 'reconstruction': X_te}),
            epochs=epochs, batch_size=batch, verbose=0,
            callbacks=get_callbacks(name)
        )
        y_pred = np.argmax(model.predict(X_te, verbose=0)[0], axis=1)
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(
            X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=epochs, batch_size=batch, verbose=0,
            callbacks=get_callbacks(name)
        )
        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    m = {
        'accuracy':  round(float(accuracy_score(y_te, y_pred)), 4),
        'f1':        round(float(f1_score(y_te, y_pred, average='weighted', zero_division=0)), 4),
        'precision': round(float(precision_score(y_te, y_pred, average='weighted', zero_division=0)), 4),
        'recall':    round(float(recall_score(y_te, y_pred, average='weighted', zero_division=0)), 4),
    }
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAMPLE STUDENTS JSON  (for Teacher dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_students(df_clean: pd.DataFrame, n: int = 120) -> list:
    """Extract n representative rows from the dataset."""
    df_sample = df_clean.sample(n=min(n, len(df_clean)), random_state=RANDOM_STATE)
    records = []
    rev_cat = {col: {v: k for k, v in mapping.items()} for col, mapping in CAT_MAP.items()}

    for i, (_, row) in enumerate(df_sample.iterrows()):
        r = {'id': f'STU{str(i+1).zfill(4)}'}
        for feat in FEATURES:
            val = row.get(feat, 0)
            if feat in rev_cat:
                val = rev_cat[feat].get(int(val), str(int(val)))
            else:
                val = float(val)
            r[feat] = val
        grade_col = next((c for c in ['Grade', 'Final_Grade'] if c in row.index), None)
        r['Grade'] = str(row.get(grade_col, '?')) if grade_col else '?'
        records.append(r)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN TRAINING ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def save_pkl(model, filename):
    if model is None: return
    joblib.dump(model, MODEL_DIR / filename)

def save_keras(model, filename):
    if model is None: return
    model.save(MODEL_DIR / filename)


def main(args):
    print("\n" + "═" * 68)
    print("  EduAI Predict — Full Training Pipeline")
    print("  Muhammad Asif Riaz  |  F22BDATS1M02032")
    print("  Islamia University of Bahawalpur")
    print("═" * 68)
    print(f"\n  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output  : {MODEL_DIR}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("▶ Step 1: Loading Dataset")
    df = load_dataset(args.data)
    X_raw, y_grade, y_risk, df_clean = preprocess(df)

    # ── Split ──────────────────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_tr_raw, X_te_raw, yg_tr, yg_te = train_test_split(
        X_raw, y_grade, test_size=0.2, random_state=RANDOM_STATE, stratify=y_grade)
    _, _, yr_tr, yr_te = train_test_split(
        X_raw, y_risk, test_size=0.2, random_state=RANDOM_STATE, stratify=y_risk)

    # ── Preprocessors ─────────────────────────────────────────────────────────
    print("\n▶ Step 2: Fitting Preprocessors")
    pkl_scaler, nn_scaler = build_preprocessors(X_tr_raw)
    grade_enc, risk_enc  = build_encoders(yg_tr, yr_tr)
    save_pkl(pkl_scaler, 'preprocessor.pkl')
    save_pkl(nn_scaler,  'nn_preprocessor.pkl')
    save_pkl(grade_enc,  'grade_encoder.pkl')
    save_pkl(risk_enc,   'risk_encoder.pkl')
    print("   ✅ preprocessor.pkl / nn_preprocessor.pkl / *_encoder.pkl saved")

    # Scaled sets
    X_tr_pkl = pkl_scaler.transform(X_tr_raw)
    X_te_pkl = pkl_scaler.transform(X_te_raw)
    X_tr_nn  = nn_scaler.transform(X_tr_raw)
    X_te_nn  = nn_scaler.transform(X_te_raw)

    # 3D views for sequence models (BiLSTM, CNN1D)
    X_tr_3d = X_tr_nn.reshape(-1, N_FEATURES, 1).astype(np.float32)
    X_te_3d = X_te_nn.reshape(-1, N_FEATURES, 1).astype(np.float32)

    # ── Feature Importance (from RF) ───────────────────────────────────────────
    fi_dict = {}

    # ── Metrics accumulator ───────────────────────────────────────────────────
    all_metrics: dict = {}

    def log(name, target, m, elapsed):
        tag = f"{name} [{target}]"
        t = m.get('type', '')
        print(f"   [{tag:38s}] acc={m.get('accuracy',0):.4f}  f1={m.get('f1',0):.4f}  "
              f"({elapsed:.1f}s)")
        key = f"{target}_{name}" if target != 'grade' else name
        all_metrics.setdefault(name, {}).update(m)

    # ═══════════════════════════════════════════════════════════════════════════
    # ── CLASSICAL ML (8) ──────────────────────────────────────────────────────
    # ═══════════════════════════════════════════════════════════════════════════
    if not args.skip_ml:
        print("\n▶ Step 3: Training Classical ML Models (8)")

        # ── Logistic Regression ──
        t0 = time.time()
        m_gr, m_info = train_logistic_regression(X_tr_raw, yg_tr, X_te_raw, yg_te, 'grade', pkl_scaler)
        m_ri, r_info = train_logistic_regression(X_tr_raw, yr_tr, X_te_raw, yr_te, 'risk',  pkl_scaler)
        save_pkl(m_gr, 'grade_logistic_regression.pkl')
        save_pkl(m_ri, 'risk_logistic_regression.pkl')
        all_metrics['logistic_regression'] = {**m_info, 'type': 'ml'}
        log('logistic_regression', 'grade', m_info, time.time()-t0)

        # ── Random Forest ──
        t0 = time.time()
        m_gr, m_info = train_random_forest(X_tr_raw, yg_tr, X_te_raw, yg_te, 'grade')
        m_ri, r_info = train_random_forest(X_tr_raw, yr_tr, X_te_raw, yr_te, 'risk')
        save_pkl(m_gr, 'grade_random_forest.pkl')
        save_pkl(m_ri, 'risk_random_forest.pkl')
        all_metrics['random_forest'] = {**m_info, 'type': 'ml'}
        log('random_forest', 'grade', m_info, time.time()-t0)
        # Extract feature importance
        if m_gr is not None and hasattr(m_gr, 'feature_importances_'):
            fi_dict = {FEATURES[i]: round(float(m_gr.feature_importances_[i]), 4)
                       for i in range(N_FEATURES)}
            fi_dict = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))

        # ── Gradient Boosting ──
        t0 = time.time()
        m_gr, m_info = train_gradient_boosting(X_tr_raw, yg_tr, X_te_raw, yg_te, 'grade')
        m_ri, r_info = train_gradient_boosting(X_tr_raw, yr_tr, X_te_raw, yr_te, 'risk')
        save_pkl(m_gr, 'grade_gradient_boosting.pkl')
        save_pkl(m_ri, 'risk_gradient_boosting.pkl')
        all_metrics['gradient_boosting'] = {**m_info, 'type': 'ml'}
        log('gradient_boosting', 'grade', m_info, time.time()-t0)

        # ── XGBoost ──
        t0 = time.time()
        m_gr, m_info = train_xgboost(X_tr_raw, yg_tr, X_te_raw, yg_te, 'grade')
        m_ri, r_info = train_xgboost(X_tr_raw, yr_tr, X_te_raw, yr_te, 'risk')
        if m_gr: save_pkl(m_gr, 'grade_xgboost.pkl')
        if m_ri: save_pkl(m_ri, 'risk_xgboost.pkl')
        if m_info: all_metrics['xgboost'] = {**m_info, 'type': 'ml'}
        if m_info: log('xgboost', 'grade', m_info, time.time()-t0)

        # ── LightGBM ──
        t0 = time.time()
        m_gr, m_info = train_lightgbm(X_tr_raw, yg_tr, X_te_raw, yg_te, 'grade')
        m_ri, r_info = train_lightgbm(X_tr_raw, yr_tr, X_te_raw, yr_te, 'risk')
        if m_gr: save_pkl(m_gr, 'grade_lightgbm.pkl')
        if m_ri: save_pkl(m_ri, 'risk_lightgbm.pkl')
        if m_info: all_metrics['lightgbm'] = {**m_info, 'type': 'ml'}
        if m_info: log('lightgbm', 'grade', m_info, time.time()-t0)

        # ── CatBoost ──
        t0 = time.time()
        m_gr, m_info = train_catboost(X_tr_raw, yg_tr, X_te_raw, yg_te, 'grade')
        m_ri, r_info = train_catboost(X_tr_raw, yr_tr, X_te_raw, yr_te, 'risk')
        if m_gr: save_pkl(m_gr, 'grade_catboost.pkl')
        if m_ri: save_pkl(m_ri, 'risk_catboost.pkl')
        if m_info: all_metrics['catboost'] = {**m_info, 'type': 'ml'}
        if m_info: log('catboost', 'grade', m_info, time.time()-t0)

        # ── Stacking ──
        print("   [stacking — training estimators, this may take 2-3 min]")
        t0 = time.time()
        m_gr, m_info = train_stacking(X_tr_raw, yg_tr, X_te_raw, yg_te, 'grade')
        m_ri, r_info = train_stacking(X_tr_raw, yr_tr, X_te_raw, yr_te, 'risk')
        save_pkl(m_gr, 'grade_stacking.pkl')
        save_pkl(m_ri, 'risk_stacking.pkl')
        all_metrics['stacking'] = {**m_info, 'type': 'ml'}
        log('stacking', 'grade', m_info, time.time()-t0)

        # ── TabNet ──
        t0 = time.time()
        m_gr, m_info = train_tabnet(X_tr_raw, yg_tr, X_te_raw, yg_te, 'grade', pkl_scaler)
        m_ri, r_info = train_tabnet(X_tr_raw, yr_tr, X_te_raw, yr_te, 'risk',  pkl_scaler)
        if m_gr: save_pkl(m_gr, 'grade_tabnet.pkl')
        if m_ri: save_pkl(m_ri, 'risk_tabnet.pkl')
        if m_info: all_metrics['tabnet'] = {**m_info, 'type': 'ml'}
        if m_info: log('tabnet', 'grade', m_info, time.time()-t0)

        print("   ✅ Classical ML training complete")

    # ═══════════════════════════════════════════════════════════════════════════
    # ── DEEP LEARNING (15) ────────────────────────────────────────────────────
    # ═══════════════════════════════════════════════════════════════════════════
    if not args.skip_dl:
        print("\n▶ Step 4: Training Deep Learning Models (15)")
        print("   (GPU will be used automatically if available)")
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            print(f"   GPUs detected: {len(gpus)} — {[g.name for g in gpus] or 'CPU only'}")
        except ImportError:
            print("   ⚠️  TensorFlow not installed — skipping Deep Learning models")
            print("       pip install tensorflow")
            args.skip_dl = True

    if not args.skip_dl:
        EP = args.epochs
        BS = args.batch

        dl_tasks = [
            # (arch_key, build_fn, grade_file, risk_file, needs_3d, dual_output)
            ('residual_nn',     build_residual_nn,     'grade_residual_nn.keras',  'risk_residual_nn.keras',   False, False),
            ('ft_transformer',  build_ft_transformer,  'grade_ft_transformer.keras','risk_tf_transformer.keras', False, False),
            ('bilstm',          build_bilstm,           'grade_bilstm.keras',       'risk_bilstm.keras',        True,  False),
            ('cnn1d',           build_cnn1d,            None,                       'risk_cnn1d.keras',         True,  False),
            ('attention_mlp',   build_attention_mlp,   None,                       'risk_attention_mlp.keras', False, False),
            ('autoencoder_clf', build_autoencoder_clf, None,                       'risk_autoencoder_clf.keras',False, True ),
            ('vae_clf',         build_vae_clf,         None,                       'risk_vae_df.keras',        False, True ),
            ('wide_and_deep',   build_wide_and_deep,   None,                       'risk_wide_and_deep.keras', False, False),
            ('swish_deep',      build_swish_deep,      None,                       'risk_swish_deep.keras',    False, False),
            ('densenet_mlp',    build_densenet_mlp,    None,                       'risk_densenet_mlp.keras',  False, False),
            ('tabformer',       build_tabformer,       None,                       'risk_tabformer.keras',     False, False),
            ('saint',           build_saint,           None,                       'risk_saint.keras',         False, False),
            ('gated_mlp',       build_gated_mlp,       None,                       'risk_gated_mlp.keras',     False, False),
            ('node',            build_node_approx,     None,                       'risk_node_approx.keras',   False, False),
            ('capsule_net',     build_capsule_net,      None,                       'risk_capsule_net.keras',   False, False),
        ]

        for (key, build_fn, g_file, r_file, needs_3d, dual) in dl_tasks:
            t0 = time.time()
            X_tr_use = X_tr_3d if needs_3d else X_tr_nn
            X_te_use = X_te_3d if needs_3d else X_te_nn

            # Train grade model if file defined
            if g_file:
                m = build_fn(N_GRADES)
                m_info = train_keras_model(m, X_tr_use, yg_tr, X_te_use, yg_te,
                                           N_GRADES, EP, BS, key, dual)
                save_keras(m, g_file)
                all_metrics.setdefault(key, {}).update({**m_info, 'type': 'deep_learning'})

            # Train risk model
            if r_file:
                m = build_fn(N_RISKS)
                m_info = train_keras_model(m, X_tr_use, yr_tr, X_te_use, yr_te,
                                           N_RISKS, EP, BS, key+'_risk', dual)
                save_keras(m, r_file)
                if key not in all_metrics:
                    all_metrics[key] = {**m_info, 'type': 'deep_learning'}

            elapsed = time.time() - t0
            m_ref = all_metrics.get(key, {})
            log(key, 'grade' if g_file else 'risk', m_ref, elapsed)

        print("   ✅ Deep Learning training complete")

    # ── Save artefacts ─────────────────────────────────────────────────────────
    print("\n▶ Step 5: Saving Artefacts")

    # Add type field to all metrics
    for k, v in all_metrics.items():
        if 'type' not in v:
            v['type'] = 'ml'

    with open(MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("   ✅ metrics.json")

    if fi_dict:
        with open(MODEL_DIR / 'feature_importance.json', 'w') as f:
            json.dump(fi_dict, f, indent=2)
        print("   ✅ feature_importance.json")

    # Sample students
    try:
        samples = generate_sample_students(df_clean, n=120)
        with open(MODEL_DIR / 'sample_students.json', 'w') as f:
            json.dump(samples, f, indent=2)
        print("   ✅ sample_students.json  (120 records)")
    except Exception as e:
        print(f"   ⚠️  Could not save sample students: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("  TRAINING SUMMARY")
    print("─" * 68)
    sorted_m = sorted(all_metrics.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)
    print(f"  {'Model':<30} {'Acc':>7}  {'F1':>7}  {'Type':>12}")
    print("  " + "─" * 62)
    for name, m in sorted_m:
        print(f"  {name:<30} {m.get('accuracy',0)*100:>6.2f}%  {m.get('f1',0)*100:>6.2f}%  {m.get('type',''):>12}")
    best = sorted_m[0] if sorted_m else ('—', {})
    print(f"\n  🏆 Best model : {best[0]}  ({best[1].get('accuracy', 0)*100:.2f}% accuracy)")
    print(f"  📁 All files  : {MODEL_DIR}")
    print(f"\n  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 68 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EduAI Predict — Train All 23 Models',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--data',     default=None, help='Path to dataset CSV/XLSX')
    parser.add_argument('--skip-dl',  action='store_true', help='Skip Deep Learning models (faster)')
    parser.add_argument('--skip-ml',  action='store_true', help='Skip Classical ML models')
    parser.add_argument('--epochs',   type=int, default=60,  help='Epochs for DL models (default: 60)')
    parser.add_argument('--batch',    type=int, default=256, help='Batch size for DL (default: 256)')
    args = parser.parse_args()
    main(args)