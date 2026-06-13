# utils.py
import json
import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
MODEL_DIR = "models"
DATA_PATH = os.path.join("data", "student_performance_grade.xlsx")

# -------------------------------------------------------------------
# 2. Load static files (metrics, feature importance)
# -------------------------------------------------------------------
def load_json(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

metrics = load_json("metrics.json")
feature_importance = load_json("feature_importance.json")

# -------------------------------------------------------------------
# 3. Load scalers
# -------------------------------------------------------------------
scaler = None
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("✅ Loaded general scaler")

nn_scaler = None
nn_scaler_path = os.path.join(MODEL_DIR, "nn_scaler.pkl")
if os.path.exists(nn_scaler_path):
    nn_scaler = joblib.load(nn_scaler_path)
    print("✅ Loaded neural network scaler")

# -------------------------------------------------------------------
# 4. Load sklearn / xgboost models
# -------------------------------------------------------------------
loaded_models = {}
for name in ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]:
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if os.path.exists(path):
        try:
            loaded_models[name] = joblib.load(path)
            print(f"✅ Loaded {name}")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")

# -------------------------------------------------------------------
# 5. Load Keras neural network
# -------------------------------------------------------------------
keras_model = None
keras_path = os.path.join(MODEL_DIR, "neural_network.keras")
if os.path.exists(keras_path):
    try:
        keras_model = tf.keras.models.load_model(keras_path)
        print("✅ Loaded neural_network.keras")
    except Exception as e:
        print(f"❌ Failed to load neural_network.keras: {e}")

# -------------------------------------------------------------------
# 6. Load dataset
# -------------------------------------------------------------------
student_df = None
if os.path.exists(DATA_PATH):
    try:
        student_df = pd.read_excel(DATA_PATH)
        print(f"✅ Loaded {len(student_df)} student records from dataset")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")

# -------------------------------------------------------------------
# 7. Preprocessing pipeline
# -------------------------------------------------------------------
CATEGORICAL_COLS = [
    "Gender", "Study_Method", "Diet_Quality",
    "Internet_Quality", "Part_Time_Job", "Family_Income_Level",
    "Extracurricular"
]
NUMERIC_COLS = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Stress_Level", "Screen_Time", "Previous_GPA",
    "Tutoring_Sessions_Per_Week", "Exam_Anxiety_Score", "Age"
]

# All feature names (from feature_importance if available)
all_feature_names = list(feature_importance.keys()) if feature_importance else None

def preprocess_input(student_dict, for_nn=False):
    """
    Transform a raw student input dict into a scaled feature vector.
    Matches the training pipeline exactly.
    """
    row = {col: float(student_dict.get(col, 0)) for col in NUMERIC_COLS}
    df = pd.DataFrame([row])
    # One‑hot encode categoricals using the expected feature names from feature_importance
    if all_feature_names:
        for cat in CATEGORICAL_COLS:
            value = student_dict.get(cat, "")
            for feature in all_feature_names:
                if feature.startswith(cat + "_") and feature.replace(cat + "_", "") == value:
                    df[feature] = 1
                elif feature.startswith(cat + "_"):
                    df[feature] = 0
        # Ensure column order matches scaler expectation
        df = df.reindex(columns=all_feature_names, fill_value=0)
    else:
        # Fallback: no feature importance file, cannot one‑hot properly
        pass

    use_scaler = nn_scaler if (for_nn and nn_scaler) else scaler
    if use_scaler:
        scaled = use_scaler.transform(df)
    else:
        scaled = df.values
    return scaled

# -------------------------------------------------------------------
# 8. Prediction functions
# -------------------------------------------------------------------
def predict_with_model(model_name, input_dict):
    """Return grade prediction using a specific model."""
    if model_name == "neural_network" and keras_model is not None:
        X = preprocess_input(input_dict, for_nn=True)
        probs = keras_model.predict(X, verbose=0)[0]
        idx = np.argmax(probs)
        grades = ['Fail', 'D', 'C', 'B', 'A']
        return grades[idx], probs.tolist(), float(np.max(probs))
    elif model_name in loaded_models:
        model = loaded_models[model_name]
        X = preprocess_input(input_dict, for_nn=False)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
        else:
            probs = model.predict(X)[0]
        idx = np.argmax(probs)
        grades = ['Fail', 'D', 'C', 'B', 'A']
        return grades[idx], probs.tolist(), float(np.max(probs))
    else:
        return None, None, None

def get_risk(probabilities):
    p_fail = probabilities[0]
    p_d = probabilities[1]
    if (p_fail + p_d) > 0.35 or p_fail > 0.05:
        return "High"
    elif (p_fail + p_d) > 0.15 or probabilities[2] > 0.5:
        return "Medium"
    return "Low"

def suggestions_for(input_dict, grade):
    tips = []
    h = float(input_dict.get("Hours_Studied", 0))
    at = float(input_dict.get("Attendance", 0))
    sl = float(input_dict.get("Sleep_Hours", 0))
    st = float(input_dict.get("Stress_Level", 0))
    sc = float(input_dict.get("Screen_Time", 0))
    gp = float(input_dict.get("Previous_GPA", 0))
    ea = float(input_dict.get("Exam_Anxiety_Score", 0))
    tu = float(input_dict.get("Tutoring_Sessions_Per_Week", 0))
    job = input_dict.get("Part_Time_Job", "No")
    diet = input_dict.get("Diet_Quality", "Average")

    if h < 3:
        tips.append({"icon": "📚", "text": f"Study time is critically low ({h:.1f} hrs/day). Target 5–7 hours daily.", "severity": "high"})
    elif h < 5:
        tips.append({"icon": "📖", "text": f"Increase daily study hours from {h:.1f} to at least 5 for better outcomes.", "severity": "medium"})
    if at < 70:
        tips.append({"icon": "🏫", "text": f"Attendance critically low ({at:.0f}%). Attend all classes immediately.", "severity": "high"})
    elif at < 80:
        tips.append({"icon": "🎒", "text": f"Attendance ({at:.0f}%) is below 80%. Aim for 85%+.", "severity": "medium"})
    if sl < 6:
        tips.append({"icon": "😴", "text": f"Sleep deprivation ({sl:.0f} hrs) impairs memory. Target 7–8 hours.", "severity": "high"})
    if st > 8:
        tips.append({"icon": "🧘", "text": f"Extreme stress ({st:.0f}/10). Seek counseling or stress-management workshops.", "severity": "high"})
    elif st > 6:
        tips.append({"icon": "🌿", "text": f"Elevated stress ({st:.0f}/10). Try exercise, mindfulness, or scheduled breaks.", "severity": "medium"})
    if sc > 6:
        tips.append({"icon": "📵", "text": f"High screen time ({sc:.0f} hrs/day) reduces study effectiveness. Limit to under 2 hours.", "severity": "medium"})
    if ea > 7:
        tips.append({"icon": "😰", "text": f"High exam anxiety ({ea:.0f}/10). Practice mock tests, breathing exercises.", "severity": "medium"})
    if tu == 0 and grade in ['C', 'D', 'Fail']:
        tips.append({"icon": "👨‍🏫", "text": "No tutoring sessions. Students with 2+ weekly sessions improve by ~1.5 grade points.", "severity": "high"})
    if gp < 2.5:
        tips.append({"icon": "📈", "text": f"Previous GPA ({gp:.2f}) is low. Focus on foundational concepts.", "severity": "medium"})
    if diet == "Poor":
        tips.append({"icon": "🥗", "text": "Poor diet reduces cognition. Add protein, complex carbs, stay hydrated.", "severity": "low"})
    if job == "Yes" and grade in ['C', 'D', 'Fail']:
        tips.append({"icon": "💼", "text": "Part-time job + poor grades is risky. Consider reducing work hours during exams.", "severity": "medium"})
    if not tips:
        tips.append({"icon": "✅", "text": "Excellent profile! Stay consistent and keep up the great work!", "severity": "low"})
    return tips[:5]

# -------------------------------------------------------------------
# 9. Chat
# -------------------------------------------------------------------
CHAT_KB = [
    (["hello","hi","hey"], "👋 Hello! I'm EduAI, your academic assistant. Ask me about grades, stress, study techniques, or anything about your performance!"),
    (["improve","better","boost","grade"], "📚 To improve grades:\n1. Study 5–7h/day with Pomodoro\n2. Attend 85%+ classes\n3. Sleep 7–8h\n4. Join 2+ tutoring sessions/week\n5. Reduce screen time to <2h\n6. Practice past papers"),
    (["stress","pressure"], "🧘 Stress tips:\n• Break tasks into small daily goals\n• Exercise 30 min daily\n• 5-4-3-2-1 grounding technique\n• Talk to a counselor\n• Avoid all-nighters"),
    (["sleep","tired"], "😴 Sleep is vital! Aim for 7-8h, consistent schedule, no screens 1h before bed, no caffeine after 4 PM."),
    (["attend"], "🏫 Attendance is a top predictor! Below 75% dramatically increases failure risk."),
    (["study method","technique"], "📖 Evidence-based:\n1. Spaced repetition\n2. Active recall\n3. Feynman technique\n4. Pomodoro (25min focus, 5min break)\n5. Practice problems > passive reading"),
    (["exam","test anxiety"], "✍️ Exam prep:\n• Start 2+ weeks early\n• Past papers under timed conditions\n• Sleep well the night before\n• Box breathing: 4s inhale, 4s hold, 6s exhale"),
    (["diet","food"], "🥗 Nutrition:\n• Eat breakfast (skipping reduces concentration 20%)\n• Omega-3: salmon, walnuts\n• Stay hydrated\n• Dark chocolate (70%+) boosts focus"),
    (["gpa"], "📊 Improving GPA: Talk to advisors, focus on understanding, complete all assignments, seek extra credit early."),
    (["tutor"], "👨‍🏫 Tutoring improves grades by ~1.5 points. University centers are often free. Use Khan Academy, MIT OCW, Coursera."),
    (["deep learning","neural network"], "🧠 22 models: 12 ML + 10 DL (MLP). Best: Logistic Regression (78.7%), Best DL: Pyramid Net (77.8%)."),
    (["predict"], "🔮 Enter your profile in the Student Panel, and all 22 models analyze it. You get grade probabilities, risk level, and AI recommendations."),
    (["thank","great"], "😊 You're welcome! Small consistent actions beat big occasional efforts. Good luck! 🎓")
]

def chat_reply(message: str) -> str:
    msg = message.lower()
    for keywords, reply in CHAT_KB:
        if any(kw in msg for kw in keywords):
            return reply
    return "🤔 Try asking about: grade improvement, stress, sleep, attendance, exam prep, diet, tutoring, or deep learning models. Use the Student Panel to get a full AI prediction!"

# -------------------------------------------------------------------
# 10. Helper for sorted model list
# -------------------------------------------------------------------
def get_sorted_models():
    avail = []
    if metrics:
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)
        for name, _ in sorted_metrics:
            if name in loaded_models or (name == "neural_network" and keras_model):
                avail.append(name)
    else:
        avail = list(loaded_models.keys())
        if keras_model:
            avail.append("neural_network")
    return avail