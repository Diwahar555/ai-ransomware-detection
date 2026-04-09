"""
live_adapt.py
Quick meta-ensemble adaptation using real-world captured data.
- Does NOT retrain the 5 base models (keeps them intact)
- Only retrains the meta-learner (LogisticRegression) on real data
- Runs in under 60 seconds
- Saves updated behavioral_meta.pkl

Run:  python live_adapt.py
"""

import os, sys, glob, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

MODELS_DIR     = "models"
REAL_WORLD_DIR = "data/behavioral/real_world"
MODEL_ORDER    = ["Random Forest", "XGBoost", "SVM", "DNN", "LSTM"]

# ── Load feature names ─────────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, "behavioral_feature_names.pkl"), "rb") as f:
    FEATURE_NAMES = pickle.load(f)

print("=" * 55)
print("  LIVE META-ENSEMBLE ADAPTATION")
print("=" * 55)
print(f"  Features  : {len(FEATURE_NAMES)}")
print(f"  Data dir  : {REAL_WORLD_DIR}")

# ── Load real-world CSVs ───────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(REAL_WORLD_DIR, "*.csv")))
if not csv_files:
    print("\n  No real-world CSVs found in", REAL_WORLD_DIR)
    print("  Run: python run_real_world_capture.py first")
    sys.exit(1)

frames = []
for path in csv_files:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        print(f"  Skipping (no label column): {os.path.basename(path)}")
        continue
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    frames.append(df)
    label_counts = df["label"].value_counts().to_dict()
    print(f"  Loaded: {os.path.basename(path):35s} rows={len(df):4d}  {label_counts}")

if not frames:
    print("  No valid CSV files found.")
    sys.exit(1)

real_df = pd.concat(frames, ignore_index=True)
print(f"\n  Total real-world rows : {len(real_df)}")
print(f"  Normal (0)            : {(real_df['label']==0).sum()}")
print(f"  Attack (1)            : {(real_df['label']==1).sum()}")

if real_df["label"].nunique() < 2:
    print("\n  Need BOTH normal (label=0) AND attack (label=1) data.")
    print("  Capture an attack session first:")
    print("  python run_real_world_capture.py --label 1 --duration 180 --session-id attack_sim_01")
    sys.exit(1)

X_real = real_df[FEATURE_NAMES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
y_real = real_df["label"].values.astype(int)

# ── Load scaler and apply ─────────────────────────────────────────────────────
print("\n-- Loading scaler and base models -------------------------")
with open(os.path.join(MODELS_DIR, "behavioral_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
X_scaled = scaler.transform(X_real)
print("  Scaler applied.")

# ── Load base models ──────────────────────────────────────────────────────────
models = {}
for name, fname in [("Random Forest", "behavioral_rf.pkl"),
                     ("XGBoost",       "behavioral_xgb.pkl"),
                     ("SVM",           "behavioral_svm.pkl")]:
    with open(os.path.join(MODELS_DIR, fname), "rb") as f:
        models[name] = pickle.load(f)
    print(f"  {fname:30s} loaded")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
for name, fname in [("DNN",  "behavioral_dnn.h5"),
                     ("LSTM", "behavioral_lstm.h5")]:
    models[name] = tf.keras.models.load_model(os.path.join(MODELS_DIR, fname), compile=False)
    print(f"  {fname:30s} loaded")

# ── Collect probabilities from all 5 base models ──────────────────────────────
print("\n-- Collecting real-world predictions from all 5 models ----")

def reshape_lstm(X, timesteps=2):
    n     = X.shape[1]
    n_per = n // timesteps
    pad   = (timesteps * n_per) - n
    if pad < 0:
        n_per += 1
        pad = (timesteps * n_per) - n
    if pad > 0:
        X = np.pad(X, ((0, 0), (0, pad)))
    return X.reshape(X.shape[0], timesteps, n_per)

meta_features = {}
for name in ["Random Forest", "XGBoost", "SVM"]:
    meta_features[name] = models[name].predict_proba(X_scaled)[:, 1]
    print(f"  {name:20s} → mean prob = {meta_features[name].mean():.3f}")

meta_features["DNN"] = models["DNN"].predict(X_scaled, verbose=0).flatten()
print(f"  {'DNN':20s} → mean prob = {meta_features['DNN'].mean():.3f}")

X_lstm = reshape_lstm(X_scaled.copy())
meta_features["LSTM"] = models["LSTM"].predict(X_lstm, verbose=0).flatten()
print(f"  {'LSTM':20s} → mean prob = {meta_features['LSTM'].mean():.3f}")

# ── Stack into meta-feature matrix ────────────────────────────────────────────
X_meta = np.column_stack([meta_features[name] for name in MODEL_ORDER])

# ── Retrain meta-ensemble on REAL data ───────────────────────────────────────
print("\n-- Retraining meta-ensemble on real-world data -----------")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
meta.fit(X_meta, y_real)

y_pred = meta.predict(X_meta)
acc    = accuracy_score(y_real, y_pred)
f1     = f1_score(y_real, y_pred, zero_division=0)

print(f"\n  Real-world adaptation results:")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  F1 Score : {f1*100:.2f}%")
print("\n" + classification_report(y_real, y_pred, target_names=["Normal","Attack"], zero_division=0))

# ── Save updated meta model ───────────────────────────────────────────────────
meta_path = os.path.join(MODELS_DIR, "behavioral_meta.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)
print(f"  Saved → {meta_path}")

# ── Also save to CSV for report ───────────────────────────────────────────────
os.makedirs("reports", exist_ok=True)
result_df = pd.DataFrame([{
    "model": "Meta-Ensemble (Real-world adapted)",
    "accuracy": round(acc, 4),
    "f1": round(f1, 4),
    "rows_used": len(real_df),
    "normal_rows": int((real_df["label"]==0).sum()),
    "attack_rows": int((real_df["label"]==1).sum()),
    "csv_files": len(csv_files),
}])
result_df.to_csv("reports/real_world_adaptation_results.csv", index=False)
print("  Saved → reports/real_world_adaptation_results.csv")

print("\n" + "=" * 55)
print("  ADAPTATION COMPLETE!")
print("  The dashboard will now use the real-world adapted model.")
print("  Restart the dashboard to load the updated meta model.")
print("=" * 55)
