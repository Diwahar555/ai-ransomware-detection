"""
src/generate_confusion_matrix.py
Update 10: Generate confusion matrices for all models and the ensemble.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

MODELS_DIR = "models"
REPORTS_DIR = "reports"
DATA_PATH = "data/behavioral/dataset_500k.csv"
MODEL_ORDER = ["Random Forest", "XGBoost", "SVM", "DNN", "LSTM"]

def load_all():
    # Load Scaler
    with open(os.path.join(MODELS_DIR, "behavioral_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    
    models = {}
    # Scikit-learn models
    for name, fname in [("Random Forest", "behavioral_rf.pkl"),
                        ("XGBoost",       "behavioral_xgb.pkl"),
                        ("SVM",           "behavioral_svm.pkl")]:
        with open(os.path.join(MODELS_DIR, fname), "rb") as f:
            models[name] = pickle.load(f)

    # Keras models
    for name, fname in [("DNN", "behavioral_dnn.h5"), ("LSTM", "behavioral_lstm.h5")]:
        models[name] = tf.keras.models.load_model(os.path.join(MODELS_DIR, fname), compile=False)

    # Meta-ensemble
    meta_path = os.path.join(MODELS_DIR, "behavioral_meta.pkl")
    ensemble_model = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            ensemble_model = pickle.load(f)
            
    return models, scaler, ensemble_model

def reshape_lstm(X, timesteps=2):
    n = X.shape[1]
    n_per = n // timesteps
    pad = (timesteps * n_per) - n
    if pad > 0:
        X = np.pad(X, ((0, 0), (0, pad)))
    return X.reshape(X.shape[0], timesteps, X.shape[1]//timesteps)

def plot_cm(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Benign", "Attack"])
    plt.yticks(tick_marks, ["Benign", "Attack"])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Run generate_500k_dataset.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    X_raw = df.drop(columns=["label"]).values
    y = df["label"].values

    # Test split (20% as requested)
    _, X_test_raw, _, y_test = train_test_split(X_raw, y, test_size=0.20, random_state=42)
    
    print("Loading models...")
    models, scaler, ensemble_model = load_all()
    X_test = scaler.transform(X_test_raw.astype(np.float32))

    all_probs = {}
    
    # Generate for each model
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, name in enumerate(MODEL_ORDER):
        print(f"Predicting with {name}...")
        if name in ["Random Forest", "XGBoost", "SVM"]:
            p = models[name].predict_proba(X_test)[:, 1]
        elif name == "DNN":
            p = models[name].predict(X_test, verbose=0).flatten()
        elif name == "LSTM":
            X_l = reshape_lstm(X_test)
            p = models[name].predict(X_l, verbose=0).flatten()
            
        all_probs[name] = p
        preds = (p >= 0.5).astype(int)
        
        cm = confusion_matrix(y_test, preds)
        save_path = os.path.join(REPORTS_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plot_cm(cm, f"CM: {name}", save_path)
        
        print(f"\n--- {name} Report ---")
        print(classification_report(y_test, preds))
        
        # Subplot
        ax = axes[i]
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(name)
        thresh = cm.max() / 2.
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="white" if cm[row, col] > thresh else "black")

    # Ensemble prediction
    print("Computing Ensemble...")
    if ensemble_model:
        meta_features = np.column_stack([all_probs[m] for m in MODEL_ORDER])
        ensemble_p = ensemble_model.predict_proba(meta_features)[:, 1]
    else:
        # Weighted avg fallback
        weights = {"Random Forest": 0.25, "XGBoost": 0.30, "SVM": 0.15, "DNN": 0.15, "LSTM": 0.15}
        ensemble_p = np.zeros_like(all_probs["DNN"])
        for m in MODEL_ORDER:
            ensemble_p += all_probs[m] * weights[m]

    ensemble_preds = (ensemble_p >= 0.5).astype(int)
    ensemble_cm = confusion_matrix(y_test, ensemble_preds)
    plot_cm(ensemble_cm, "CM: Meta-Ensemble", os.path.join(REPORTS_DIR, "confusion_matrix_ensemble.png"))
    
    print("\n--- Meta-Ensemble Report ---")
    print(classification_report(y_test, ensemble_preds))
    
    ax = axes[5]
    ax.imshow(ensemble_cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Meta-Ensemble")
    thresh = ensemble_cm.max() / 2.
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(ensemble_cm[row, col]), ha="center", va="center", color="white" if ensemble_cm[row, col] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrices_all.png"))
    print(f"\nAll confusion matrices saved to {REPORTS_DIR}")

if __name__ == "__main__":
    main()
