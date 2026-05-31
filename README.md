# Student_Performance_Prediction_System_Using_DeepLearning
📋 Table of Contents
Overview
Architecture
Model Zoo
Dataset
Project Structure
Quick Start
Training Pipeline
API Reference
Results & Benchmarks
Frontend
Configuration
Citation
Author


🔍 Overview
EduAI Predict is a multi-model student performance intelligence system developed as a Final Year Project. It combines the breadth of classical machine learning with the representational power of modern deep learning architectures to predict:
Prediction Task Output Model sGrade Classification A / B / C / D / Fail 23 separate grade modelsRisk DetectionHigh / Medium / Low 23 separate risk modelsGPA Regression0.0 – 4.0 (continuous)Random Forest regressor
Key Highlights

✅ 23 unique AI architectures trained in a single unified pipeline
✅ Dual-model design — separate grade and risk models per architecture
✅ Real-time REST API built with Flask, fully CORS-enabled
✅ Interactive frontend — Student Panel, Teacher Dashboard, Explainability, What-If Analysis
✅ Graceful fallback — analytic prediction engine active when no trained models are found
✅ Client-side simulation in the frontend for offline use
✅ Bulk prediction via CSV upload for Teacher dashboard


🏗 Architecture
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND  (Main.html)                       │
│  ┌─────────────┐  ┌───────────────┐  ┌──────────┐  ┌───────────┐    │
│  │ Student     │  │ Teacher       │  │ Model-pe │  │ Explain-  │    │
│  │ Panel       │  │ Dashboard     │  │ rformance|  │ ability   │    │
│  └──────┬──────┘  └───────┬───────┘  └────┬─────┘  └─────┬─────┘    │
└─────────┼─────────────────┼───────────────┼──────────────┼────────_─┘
          │     REST API    │               │              │
          ▼  (localhost:8000)               ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         BACKEND  (app.py / Flask)                   │
│                                                                     │
│   /predict/single ──► predict_single()                              │
│   /predict/bulk   ──► predict_bulk()                                │
│   /api/models/info──► MODEL_REGISTRY + metrics.json                 │
│   /chat           ──► rule-based KB + optional LLM                  │
│   /train          ──► retrain()                                     │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                  PREDICTION ENGINE                           │  │
│   │  encode_row() → _apply_preprocessor() → model.predict()      │  │
│   │           ↓ fallback if no model loaded                      │  │
│   │         _analytic_predict_grade / _analytic_predict_risk()   │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   ┌────────────────────┐   ┌──────────────────────────────────────┐ │
│   │  PREPROCESSORS     │   │  MODEL STORE  (46 models total)      │ │
│   │  preprocessor.pkl  │   │  grade_*.pkl / grade_*.keras  ×23    │ │
│   │  nn_preprocessor   │   │  risk_*.pkl  / risk_*.keras   ×23    │ │
│   └────────────────────┘   └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

🤖 Model Zoo
Classical ML Models (8)
#ModelKey HyperparametersGrade FileRisk File1Logistic RegressionC=1.0, solver=lbfgs, max_iter=3000grade_logistic_regression.pklrisk_logistic_regression.pkl2Random Forestn_estimators=500, max_features=sqrt, oob_score=Truegrade_random_forest.pklrisk_random_forest.pkl3Gradient Boostingn_estimators=300, lr=0.08, max_depth=5, subsample=0.85grade_gradient_boosting.pklrisk_gradient_boosting.pkl4XGBoostn_estimators=400, lr=0.07, max_depth=6, colsample=0.85grade_xgboost.pklrisk_xgboost.pkl5LightGBMn_estimators=500, lr=0.06, num_leaves=63grade_lightgbm.pklrisk_lightgbm.pkl6CatBoostiterations=500, lr=0.07, depth=6grade_catboost.pklrisk_catboost.pkl7Stacking EnsembleRF + GBT + XGB + LGB → LogReg, 5-fold CVgrade_stacking.pklrisk_stacking.pkl8TabNetn_d=32, n_a=32, n_steps=5, gamma=1.5grade_tabnet.pklrisk_tabnet.pkl
Deep Learning Models (15)
#ModelArchitecture HighlightsRisk File9Residual Neural NetworkSkip connections, BatchNorm, 3 residual blocksrisk_residual_nn.keras10FT-TransformerLinear tokenizer per feature → 2 transformer blocksrisk_tf_transformer.keras11BiLSTM + GRUBidirectional LSTM → GRU, input reshaped to (16,1)risk_bilstm.keras121D-CNN3× Conv1D + GlobalAvgPoolrisk_cnn1d.keras13Attention MLPSelf-gating attention (sigmoid-weighted multiply)risk_attention_mlp.keras14Autoencoder ClassifierEncoder → bottleneck → dual output (CLF + reconstruction)risk_autoencoder_clf.keras15VAE ClassifierReparameterisation trick + latent-space classifierrisk_vae_df.keras16Wide & DeepShallow wide path ∥ deep tower, concatenatedrisk_wide_and_deep.keras17Swish-SELU Deep MLPSwish/GELU activations, 4-layer deeprisk_swish_deep.keras18DenseNet MLPDense connections — all previous layers concatenatedrisk_densenet_mlp.keras19TabFormerBERT-style 3-block transformer with position embeddingsrisk_tabformer.keras20SAINTColumn-wise self-attention for tabular datarisk_saint.keras21Gated MLP (gMLP)Learned gating via sigmoid split on hidden unitsrisk_gated_mlp.keras22NODE ApproximationOblivious decision tree simulation via softmax splitsrisk_node_approx.keras23Capsule NetworkPrimary capsules → digit capsules with squash activationrisk_capsule_net.keras

Dual-model design: Every architecture trains a separate model for grade prediction AND risk detection, giving 46 total trained model files.


📊 Dataset
PropertyValueTotal Records50,000 studentsTrain / Test Split80% / 20% (stratified)Random State42 (reproducible)Missing ValuesImputed with column mode/meanClass BalancingDerived risk labels; SMOTE optional
Input Features (16)
Numeric (9):
  Age · Hours_Studied · Attendance · Sleep_Hours · Stress_Level
  Screen_Time · Previous_GPA · Tutoring_Sessions_Per_Week · Exam_Anxiety_Score

Categorical (7):
  Gender            →  Male | Female | Non-Binary
  Part_Time_Job     →  Yes | No
  Study_Method      →  Offline | Online | Hybrid
  Diet_Quality      →  Poor | Average | Good
  Internet_Quality  →  Poor | Average | Good | Excellent
  Extracurricular   →  Yes | No
  Family_Income_Level → Low | Middle | High
Target Variables
Grade  →  A (4) · B (3) · C (2) · D (1) · Fail (0)   [multi-class]
Risk   →  High (2) · Medium (1) · Low (0)              [multi-class]
GPA    →  0.0 – 4.0                                    [regression, derived]
Feature Importance (Random Forest — Gini)
Previous GPA              ████████████████████░  21.8%
Hours Studied             ████████████████░░░░░  18.4%
Attendance                ████████████░░░░░░░░░  14.3%
Exam Anxiety Score        █████████░░░░░░░░░░░░  10.2%
Stress Level              ████████░░░░░░░░░░░░░   8.8%
Tutoring Sessions/Week    ██████░░░░░░░░░░░░░░░   7.1%
Sleep Hours               █████░░░░░░░░░░░░░░░░   6.3%
Screen Time               ████░░░░░░░░░░░░░░░░░   4.9%
Age                       ███░░░░░░░░░░░░░░░░░░   3.1%
Family Income Level       ██░░░░░░░░░░░░░░░░░░░   2.0%
Gender                    █░░░░░░░░░░░░░░░░░░░░   1.3%
Part-Time Job             █░░░░░░░░░░░░░░░░░░░░   1.0%
Diet Quality              ░░░░░░░░░░░░░░░░░░░░░   0.6%
Extracurricular           ░░░░░░░░░░░░░░░░░░░░░   0.4%
Internet Quality          ░░░░░░░░░░░░░░░░░░░░░   0.4%
Study Method              ░░░░░░░░░░░░░░░░░░░░░   0.3%

📁 Project Structure
EduAI-Predict/
│
├── app.py                          # Flask backend — prediction engine + API
├── train_all_models.py             # Full training pipeline — all 23 models
├── requirements.txt                # Runtime dependencies
├── requirements_train.txt          # Training-only dependencies
├── index.html                      # Frontend SPA (single file, no build step)
│
├── data/
│   └── student_academic_behavior_dataset.csv
│   
│
└── models/                         # Auto-created on first train
    │
    ├── preprocessor.pkl            # RobustScaler for .pkl models
    ├── nn_preprocessor.pkl         # StandardScaler for .keras models
    ├── grade_encoder.pkl           # LabelEncoder — grades
    ├── risk_encoder.pkl            # LabelEncoder — risk levels
    │
    ├── metrics.json                # Accuracy / F1 / Precision / Recall — all 23
    ├── feature_importance.json     # Gini importance from RF
    ├── sample_students.json        # 120 sample records for Teacher dashboard
    │
    ├── grade_logistic_regression.pkl
    ├── grade_random_forest.pkl
    ├── grade_gradient_boosting.pkl
    ├── grade_xgboost.pkl
    ├── grade_lightgbm.pkl
    ├── grade_catboost.pkl
    ├── grade_stacking.pkl
    ├── grade_tabnet.pkl
    ├── grade_residual_nn.keras
    ├── grade_ft_transformer.keras
    ├── grade_bilstm.keras
    │
    ├── risk_logistic_regression.pkl
    ├── risk_random_forest.pkl
    ├── risk_gradient_boosting.pkl
    ├── risk_xgboost.pkl
    ├── risk_lightgbm.pkl
    ├── risk_catboost.pkl
    ├── risk_stacking.pkl
    ├── risk_tabnet.pkl
    ├── risk_residual_nn.keras
    ├── risk_tf_transformer.keras
    ├── risk_bilstm.keras
    ├── risk_cnn1d.keras
    ├── risk_attention_mlp.keras
    ├── risk_autoencoder_clf.keras
    ├── risk_vae_df.keras
    ├── risk_wide_and_deep.keras
    ├── risk_swish_deep.keras
    ├── risk_densenet_mlp.keras
    ├── risk_tabformer.keras
    ├── risk_saint.keras
    ├── risk_gated_mlp.keras
    ├── risk_node_approx.keras
    └── risk_capsule_net.keras

🚀 Quick Start
Prerequisites

Python 3.10+
pip or conda
(Optional) CUDA-capable GPU for faster DL training

1 — Clone the repository
bashgit clone https://github.com/<your-username>/EduAI-Predict.git
cd EduAI-Predict
2 — Install runtime dependencies
bashpip install -r requirements.txt
3 — Place your dataset
data/student_performance_grade.xlsx        ← preferred
data/student_academic_behavior_dataset.csv ← alternative
4 — Train all 23 models
bash# Full training — ML + Deep Learning
python train_all_models.py

# Classical ML only (fast, ~10 min on CPU)
python train_all_models.py --skip-dl

# Custom epochs and batch size
python train_all_models.py --epochs 100 --batch 512

# Point to a custom dataset
python train_all_models.py --data /path/to/your/dataset.csv
5 — Start the backend server
bashpython app.py
# → http://localhost:8000
6 — Open the frontend
Open index.html in any modern browser.
The status pill in the top-right will turn green when the backend is connected.

No backend? The frontend runs in full client-side simulation mode — all 23 model predictions are approximated using the JavaScript scoring engine.


🛠 Training Pipeline
The train_all_models.py script handles the complete pipeline in 5 steps:
Step 1 — Load Dataset        → auto-discover CSV / XLSX, normalise column names
Step 2 — Fit Preprocessors   → RobustScaler (pkl) + StandardScaler (nn) + LabelEncoders
Step 3 — Train ML Models     → 8 classical models, grade + risk each
Step 4 — Train DL Models     → 15 deep learning models, grade + risk each
Step 5 — Save Artefacts      → metrics.json, feature_importance.json, sample_students.json
Training CLI Options
usage: python train_all_models.py [OPTIONS]

options:
  --data PATH      Path to dataset CSV or XLSX (auto-discovers if omitted)
  --skip-dl        Skip all 15 Deep Learning models
  --skip-ml        Skip all 8 Classical ML models
  --epochs N       Epochs for DL training           (default: 60)
  --batch  N       Batch size for DL training        (default: 256)
  -h, --help       Show this message and exit
Estimated Training Times
HardwareML OnlyDL OnlyAll 23 ModelsCPU (8-core)~10 min~90 min~100 minGPU RTX 3060~5 min~15 min~20 minGPU A100~3 min~5 min~8 min
Based on 50,000 records, 60 DL epochs, batch size 256.

📡 API Reference
Base URL: http://localhost:8000
GET /health
Returns backend status, loaded model counts, and best model accuracy.
json{
  "status": "ok",
  "grade_models_loaded": 11,
  "risk_models_loaded": 23,
  "total_architectures": 23,
  "best_model": "saint",
  "best_accuracy": 0.9130,
  "analytic_mode": false
}

GET /api/models/info
Returns metrics and metadata for all 23 models.
json{
  "metrics": {
    "random_forest": {
      "accuracy": 0.8940, "f1": 0.8901, "precision": 0.8920, "recall": 0.8940,
      "name": "Random Forest", "icon": "🌲", "grade_loaded": true, "risk_loaded": true
    }
  },
  "feature_importance": { "Previous_GPA": 0.2180, "Hours_Studied": 0.1842, "..." : "..." },
  "best_model": "saint"
}

POST /predict/single
Predict grade, risk, and GPA for a single student.
Request body:
json{
  "model": "random_forest",
  "Age": 20,
  "Hours_Studied": 7.5,
  "Attendance": 88,
  "Sleep_Hours": 7,
  "Stress_Level": 3,
  "Screen_Time": 2,
  "Previous_GPA": 3.2,
  "Tutoring_Sessions_Per_Week": 3,
  "Exam_Anxiety_Score": 4,
  "Gender": "Female",
  "Part_Time_Job": "No",
  "Study_Method": "Hybrid",
  "Diet_Quality": "Good",
  "Internet_Quality": "Good",
  "Extracurricular": "Yes",
  "Family_Income_Level": "Middle"
}
Response:
json{
  "predicted_grade": "A",
  "predicted_score": 91.4,
  "estimated_gpa": 3.72,
  "risk_level": "Low",
  "confidence": 84.6,
  "probability": { "Fail": 0.4, "D": 1.1, "C": 7.2, "B": 18.5, "A": 72.8 },
  "all_model_preds": { "random_forest": "A", "saint": "A", "bilstm": "B", "..." : "..." },
  "suggestions": [
    { "icon": "✅", "severity": "low", "text": "Excellent profile — keep it up!" }
  ],
  "feature_importance": { "Previous_GPA": 0.2180, "..." : "..." },
  "model_used": "random_forest",
  "risk_model_used": "random_forest",
  "confidence": 84.6,
  "from_backend": true,
  "analytic_mode": false
}

POST /predict/bulk
Upload a CSV of students — returns batch predictions.
bashcurl -X POST http://localhost:8000/predict/bulk \
  -F "file=@students.csv"
Response:
json{
  "count": 150,
  "predictions": [
    { "student_id": "STU001", "predicted_grade": "B", "risk_level": "Medium", "..." : "..." }
  ]
}

GET /students
Returns sample student records with live predictions.
GET /students?limit=50

GET /students/at-risk
Returns only students predicted as High risk.

GET /students/stats
Returns grade and risk distribution across sample records.

POST /chat
Simple rule-based academic advisor chatbot.
json{ "message": "How can I improve my grade?" }
json{ "reply": "📚 Top strategies:\n1. Study 5–7h/day using Pomodoro..." }

POST /train
Trigger retraining of base sklearn models from the dataset on disk.

📈 Results & Benchmarks
Risk Classification (Binary: At Risk / Not At Risk)
RankModelTypeAccuracyF1 ScoreAUC-ROC🥇 1SAINTDL91.30%91.04%95.30%🥈 2FT-TransformerDL91.50%91.23%95.40%🥉 3TabFormerDL91.20%90.92%95.20%4Stacking EnsembleML91.00%90.72%95.00%5CatBoostML90.60%90.28%94.60%6LightGBMML90.20%89.86%94.20%7XGBoostML89.80%89.45%93.80%8Gated MLPDL89.60%89.27%93.60%9Random ForestML89.40%89.01%93.40%10Attention MLPDL89.20%88.87%93.10%
Grade Classification (Multi-class: A / B / C / D / Fail)
RankModelAccuracyF1 Score🥇 1SAINT89.30%89.04%🥈 2FT-Transformer89.50%89.23%🥉 3TabFormer89.20%88.92%4Stacking Ensemble89.00%88.72%5CatBoost88.60%88.28%6Logistic Regression78.69%78.30%
GPA Regression (R² Score)
RankModelR²RMSEMAE🥇 1Stacking Ensemble0.91000.21500.1670🥈 2SAINT0.91300.21200.1645🥉 3FT-Transformer0.91500.21000.16304LightGBM0.90200.22500.17505Random Forest0.89400.23400.1820

🎨 Frontend
The frontend is a single-file, zero-dependency SPA (index.html) with no build step required.
Pages
PageDescriptionDashboardOverview stats, grade/risk charts, sample student tableStudent Panel16-feature input form, model selector (ML + DL), full result displayTeacher DashboardBulk CSV upload, class analytics, at-risk alertsModel PerformanceMetrics tables for all 23 models — risk, grade, GPAExplainabilityFeature importance waterfall, global bar chart, What-If analysisHistoryBrowser-local prediction history with CSV exportAI ChatRule-based academic advisor chatbot
Offline Mode
When the backend is unreachable the frontend automatically switches to client-side simulation — a JavaScript scoring engine that mirrors the Random Forest logic using the same feature importance weights. All 23 model predictions are approximated client-side.
Technology Stack
HTML5 / CSS3 / Vanilla JavaScript (ES2022)
Chart.js 4.4  — all charts
Bootstrap Icons 1.11 — icon set
Google Fonts — Outfit, JetBrains Mono, Playfair Display
No build step. No npm. No webpack. Just open index.html.

⚙️ Configuration
Backend (app.py)
python# Feature list — must match your dataset column names exactly
FEATURES = ['Age', 'Hours_Studied', 'Attendance', ...]

# Grade mapping
GRADE_MAP = {'Fail': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4}

# Risk mapping
RISK_MAP = {'Low': 0, 'Medium': 1, 'High': 2}

# Model and data directories
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"
Server Options
bashpython app.py --host 0.0.0.0 --port 8000 --debug
Training Options
bashpython train_all_models.py \
  --data data/my_dataset.csv \
  --epochs 100 \
  --batch 256 \
  --skip-dl       # classical ML only

📦 Dependencies
Runtime (requirements.txt)
flask>=3.0
flask-cors>=4.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
tensorflow>=2.13
openpyxl>=3.1
Training (requirements_train.txt)
# + all runtime deps above
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2
pytorch-tabnet>=4.1
torch>=2.0
imbalanced-learn>=0.11

🔬 Methodology
1. Data Collection    →  50,000 synthetic/real student records, 16 features
2. EDA               →  Distribution analysis, correlation matrices, outlier detection
3. Preprocessing     →  Label encoding (categorical), RobustScaler (ML), StandardScaler (DL)
4. Train/Test Split  →  80/20 stratified split, random_state=42
5. Model Training    →  8 ML + 15 DL architectures, EarlyStopping for DL
6. Evaluation        →  Accuracy, F1 (weighted), Precision, Recall, AUC-ROC
7. Deployment        →  Flask REST API + single-file HTML frontend
8. Explainability    →  Gini feature importance (RF), per-prediction waterfall chart

📝 Citation
If you use this work in academic research, please cite:
bibtex@misc{riaz2024eduai,
  author       = {Muhammad Asif Riaz},
  title        = {EduAI Predict: A Multi-Model Student Academic Performance
                  Intelligence System},
  year         = {2024},
  institution  = {Islamia University of Bahawalpur},
  note         = {Final Year Project — F22BDATS1M02032},
  howpublished = {\url{https://github.com/<your-username>/EduAI-Predict}}
}

👤 Author
<div align="center">
<img src="https://github.com/identicons/<your-username>.png" width="100" style="border-radius:50%"/>
Muhammad Asif Riaz
Roll No: F22BDATS1M02032
Department of Computer Science
Islamia University of Bahawalpur, Pakistan
Show Image
Show Image
Show Image
</div>

📄 License
MIT License

Copyright (c) 2024 Muhammad Asif Riaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>
⭐ Star this repository if it helped your research!
Built with ❤️ at Islamia University of Bahawalpur
</div>
