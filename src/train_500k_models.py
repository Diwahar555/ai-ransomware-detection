"""
train_500k_models.py
Train all 5 models on the 1000,000 sample dataset.
Run from project root:
    python run_500k_training.py
"""

import os
import glob
import json
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")

DATA_PATH   = "data/behavioral/dataset_500k.csv"
MODELS_DIR  = "models"
FIGURES_DIR = "reports/figures"
REAL_WORLD_DIR = "data/behavioral/real_world"
LOGS_DIR    = "logs"
PROGRESS_LOG = os.path.join(LOGS_DIR, "training_progress.json")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class TrainingLogger:
    MODELS = ["Random Forest", "XGBoost", "SVM", "DNN", "LSTM", "Meta-Ensemble"]

    def __init__(self):
        self._state = {
            "status": "starting",
            "started_at": time.time(),
            "updated_at": time.time(),
            "current_model": None,
            "current_model_index": 0,
            "total_models": len(self.MODELS),
            "models": {m: {"status": "pending", "metrics": {}} for m in self.MODELS},
            "log": [],
            "results": [],
        }
        self._flush()

    def _flush(self):
        self._state["updated_at"] = time.time()
        try:
            with open(PROGRESS_LOG, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception:
            pass

    def start_model(self, name):
        idx = self.MODELS.index(name) if name in self.MODELS else 0
        self._state["current_model"] = name
        self._state["current_model_index"] = idx + 1
        self._state["status"] = "training"
        self._state["models"][name]["status"] = "training"
        self._state["models"][name]["started_at"] = time.time()
        self._log(f"▶ Starting {name}")
        self._flush()

    def finish_model(self, name, metrics: dict):
        self._state["models"][name]["status"] = "done"
        self._state["models"][name]["metrics"] = metrics
        self._state["models"][name]["elapsed"] = round(
            time.time() - self._state["models"][name].get("started_at", time.time()), 1
        )
        self._state["results"].append({"model": name, **metrics})
        self._log(f"✔ Finished {name} — F1: {metrics.get('f1', 0)*100:.2f}%  AUC: {metrics.get('auc', 0)*100:.2f}%")
        self._flush()

    def log(self, msg):
        self._log(msg)
        self._flush()

    def _log(self, msg):
        entry = {"t": round(time.time() - self._state["started_at"], 1), "msg": msg}
        self._state["log"].append(entry)
        if len(self._state["log"]) > 200:
            self._state["log"] = self._state["log"][-200:]

    def done(self):
        self._state["status"] = "done"
        self._state["current_model"] = None
        self._log("🎉 All models trained successfully!")
        self._flush()

    def error(self, msg):
        self._state["status"] = "error"
        self._log(f"❌ ERROR: {msg}")
        self._flush()


LOGGER = TrainingLogger()


def load_real_world_data(feature_names):
    if not os.path.isdir(REAL_WORLD_DIR):
        return pd.DataFrame()

    frames = []
    for path in sorted(glob.glob(os.path.join(REAL_WORLD_DIR, "*.csv"))):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  Skipping real-world file {path}: {exc}")
            continue

        if "label" not in df.columns:
            print(f"  Skipping real-world file without label column: {path}")
            continue

        base = os.path.splitext(os.path.basename(path))[0]
        session_values = df["session_id"] if "session_id" in df.columns else pd.Series([base] * len(df))
        host_values = df["host_id"] if "host_id" in df.columns else pd.Series([base] * len(df))

        trimmed = pd.DataFrame()
        for feat in feature_names:
            trimmed[feat] = pd.to_numeric(df.get(feat, 0), errors="coerce").fillna(0.0)
        trimmed["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        trimmed["session_id"] = session_values.astype(str).fillna(base)
        trimmed["host_id"] = host_values.astype(str).fillna(base)
        frames.append(trimmed)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def split_real_world_data(real_world_df):
    if real_world_df.empty:
        return real_world_df, real_world_df, real_world_df, {}

    df = real_world_df.copy().reset_index(drop=True)
    summary = {
        "rows": len(df),
        "sessions": int(df["session_id"].nunique()),
        "hosts": int(df["host_id"].nunique()),
    }

    def empty_like():
        return df.iloc[0:0].copy()

    if summary["sessions"] >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        adapt_idx, eval_idx = next(splitter.split(df, df["label"], groups=df["session_id"]))
        adapt_df = df.iloc[adapt_idx].reset_index(drop=True)
        eval_df = df.iloc[eval_idx].reset_index(drop=True)

        if adapt_df["session_id"].nunique() >= 2 and len(adapt_df) >= 12:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            train_idx, val_idx = next(
                splitter.split(adapt_df, adapt_df["label"], groups=adapt_df["session_id"])
            )
            train_df = adapt_df.iloc[train_idx].reset_index(drop=True)
            val_df = adapt_df.iloc[val_idx].reset_index(drop=True)
        elif len(adapt_df) >= 10 and adapt_df["label"].nunique() > 1:
            train_df, val_df = train_test_split(
                adapt_df,
                test_size=0.25,
                random_state=42,
                stratify=adapt_df["label"],
            )
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        else:
            train_df = adapt_df
            val_df = empty_like()
    elif len(df) >= 30 and df["label"].nunique() > 1:
        train_df, temp_df = train_test_split(
            df,
            test_size=0.4,
            random_state=42,
            stratify=df["label"],
        )
        val_df, eval_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df["label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)
    else:
        train_df = df.reset_index(drop=True)
        val_df = empty_like()
        eval_df = empty_like()

    summary["train_rows"] = len(train_df)
    summary["val_rows"] = len(val_df)
    summary["eval_rows"] = len(eval_df)
    return train_df, val_df, eval_df, summary


def load_data():
    print("-- Loading 500K dataset ------------------------------")
    print("  (This may take 30-60 seconds for 500K rows)")

    with open(os.path.join(MODELS_DIR, "behavioral_feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded: {len(df):,} rows x {len(feature_names)} features")
    print(f"  Normal : {(df['label']==0).sum():,}  Attack: {(df['label']==1).sum():,}")

    X = df[feature_names].values.astype(np.float32)
    y = df["label"].values.astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    external_eval = None
    real_world_df = load_real_world_data(feature_names)
    if not real_world_df.empty:
        rw_train_df, rw_val_df, rw_eval_df, rw_summary = split_real_world_data(real_world_df)
        print(
            "  Real-world data:"
            f" rows={rw_summary['rows']:,} sessions={rw_summary['sessions']} hosts={rw_summary['hosts']}"
        )
        print(
            f"    train={rw_summary['train_rows']:,} val={rw_summary['val_rows']:,}"
            f" external_eval={rw_summary['eval_rows']:,}"
        )

        if not rw_train_df.empty:
            X_train = np.vstack([X_train, rw_train_df[feature_names].values.astype(np.float32)])
            y_train = np.concatenate([y_train, rw_train_df["label"].values.astype(int)])
        if not rw_val_df.empty:
            X_val = np.vstack([X_val, rw_val_df[feature_names].values.astype(np.float32)])
            y_val = np.concatenate([y_val, rw_val_df["label"].values.astype(int)])
        if not rw_eval_df.empty:
            external_eval = {
                "name": "Real-world holdout",
                "X": rw_eval_df[feature_names].values.astype(np.float32),
                "y": rw_eval_df["label"].values.astype(int),
                "rows": len(rw_eval_df),
                "sessions": int(rw_eval_df["session_id"].nunique()),
                "hosts": int(rw_eval_df["host_id"].nunique()),
            }
    else:
        print("  Real-world data: none found in data/behavioral/real_world")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    if external_eval is not None:
        external_eval["X"] = scaler.transform(external_eval["X"])

    with open(os.path.join(MODELS_DIR, "behavioral_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n  Train : {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, external_eval


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def evaluate(name, y_true, y_pred, y_prob=None, evaluation_dataset="Synthetic holdout"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = safe_auc(y_true, y_prob) if y_prob is not None else np.nan
    print(f"\n  {name}:")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {prec*100:.2f}%")
    print(f"    Recall    : {rec*100:.2f}%")
    print(f"    F1        : {f1*100:.2f}%")
    auc_text = "N/A" if np.isnan(auc) else f"{auc*100:.2f}%"
    print(f"    AUC       : {auc_text}")
    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc,
            "y_pred": y_pred, "y_prob": y_prob, "y_true": y_true,
            "evaluation_dataset": evaluation_dataset}


def reshape_lstm(X, timesteps=2):
    n = X.shape[1]
    n_per = n // timesteps
    pad = (timesteps * n_per) - n
    if pad < 0:
        n_per += 1
        pad = (timesteps * n_per) - n
    if pad > 0:
        X = np.pad(X, ((0, 0), (0, pad)))
    return X.reshape(X.shape[0], timesteps, n_per)


def metric_summary(name, y_true, y_prob, evaluation_dataset="Synthetic holdout"):
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": safe_auc(y_true, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_true": y_true,
        "evaluation_dataset": evaluation_dataset,
    }


def collect_model_probabilities(model_bundle, X):
    probs = {
        "Random Forest": model_bundle["Random Forest"].predict_proba(X)[:, 1],
        "XGBoost": model_bundle["XGBoost"].predict_proba(X)[:, 1],
        "SVM": model_bundle["SVM"].predict_proba(X)[:, 1],
        "DNN": model_bundle["DNN"].predict(X, verbose=0).flatten(),
    }
    probs["LSTM"] = model_bundle["LSTM"].predict(reshape_lstm(X), verbose=0).flatten()
    return probs


def train_meta_ensemble(model_bundle, X_val, y_val, X_test, y_test, external_eval=None):
    model_order = ["Random Forest", "XGBoost", "SVM", "DNN", "LSTM"]
    val_probs = collect_model_probabilities(model_bundle, X_val)
    test_probs = collect_model_probabilities(model_bundle, X_test)

    meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    meta.fit(np.column_stack([val_probs[name] for name in model_order]), y_val)

    with open(os.path.join(MODELS_DIR, "behavioral_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("  Saved -> models/behavioral_meta.pkl")

    test_meta_prob = meta.predict_proba(
        np.column_stack([test_probs[name] for name in model_order])
    )[:, 1]
    ensemble_result = evaluate(
        "Calibrated Ensemble",
        y_test,
        (test_meta_prob >= 0.5).astype(int),
        test_meta_prob,
        evaluation_dataset="Synthetic holdout",
    )

    if external_eval is not None and len(external_eval["y"]) > 0:
        ext_probs = collect_model_probabilities(model_bundle, external_eval["X"])
        external_rows = []
        for name in model_order:
            external_rows.append(
                metric_summary(name, external_eval["y"], ext_probs[name], external_eval["name"])
            )
        ext_meta_prob = meta.predict_proba(
            np.column_stack([ext_probs[name] for name in model_order])
        )[:, 1]
        external_rows.append(
            metric_summary("Calibrated Ensemble", external_eval["y"], ext_meta_prob, external_eval["name"])
        )
        ext_df = pd.DataFrame([
            {k: v for k, v in row.items() if k not in ["y_pred", "y_prob", "y_true"]}
            for row in external_rows
        ])
        ext_df.to_csv("reports/real_world_eval_results.csv", index=False)
        print("  Saved -> reports/real_world_eval_results.csv")

    return ensemble_result


def train_all(X_train, X_val, X_test, y_train, y_val, y_test, external_eval=None, logger=None):
    results = []
    if logger is None:
        logger = LOGGER

    # ── Model 1: Random Forest ────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  MODEL 1: RANDOM FOREST")
    print("="*55)
    logger.start_model("Random Forest")
    print("  Training on 500K samples (may take 5-10 min)...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=10,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=1
    )
    rf.fit(X_train, y_train)
    with open(os.path.join(MODELS_DIR, "behavioral_rf.pkl"), "wb") as f:
        pickle.dump(rf, f)
    print("\n  Saved -> models/behavioral_rf.pkl")
    r_rf = evaluate("Random Forest", y_test,
                    rf.predict(X_test),
                    rf.predict_proba(X_test)[:, 1])
    logger.finish_model("Random Forest", {k: float(v) for k, v in r_rf.items() if k in ("accuracy","precision","recall","f1","auc")})
    results.append(r_rf)

    # ── Model 2: XGBoost ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  MODEL 2: XGBOOST")
    print("="*55)
    logger.start_model("XGBoost")
    print("  Training (may take 5-8 min)...")
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, verbosity=0, n_jobs=-1, tree_method="hist"
    )
    xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    with open(os.path.join(MODELS_DIR, "behavioral_xgb.pkl"), "wb") as f:
        pickle.dump(xgb_m, f)
    print("  Saved -> models/behavioral_xgb.pkl")
    r_xgb = evaluate("XGBoost", y_test,
                     xgb_m.predict(X_test),
                     xgb_m.predict_proba(X_test)[:, 1])
    logger.finish_model("XGBoost", {k: float(v) for k, v in r_xgb.items() if k in ("accuracy","precision","recall","f1","auc")})
    results.append(r_xgb)

    # ── Model 3: SVM with calibration (FIXED) ────────────────────────────────
    print("\n" + "="*55)
    print("  MODEL 3: SVM (calibrated LinearSVC)")
    print("="*55)
    logger.start_model("SVM")
    idx = np.random.choice(len(X_train), min(30000, len(X_train)), replace=False)
    svm_base = LinearSVC(C=1.0, class_weight="balanced",
                         random_state=42, max_iter=5000)
    svm = CalibratedClassifierCV(svm_base, cv=3, method="sigmoid")
    print("  Training calibrated SVM on 30K samples...")
    svm.fit(X_train[idx], y_train[idx])
    with open(os.path.join(MODELS_DIR, "behavioral_svm.pkl"), "wb") as f:
        pickle.dump(svm, f)
    print("  Saved -> models/behavioral_svm.pkl")
    r_svm = evaluate("SVM", y_test,
                     svm.predict(X_test),
                     svm.predict_proba(X_test)[:, 1])
    logger.finish_model("SVM", {k: float(v) for k, v in r_svm.items() if k in ("accuracy","precision","recall","f1","auc")})
    results.append(r_svm)

    # ── Model 4: DNN ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  MODEL 4: DEEP NEURAL NETWORK")
    print("="*55)
    logger.start_model("DNN")
    dnn = Sequential([
        Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        BatchNormalization(), Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(), Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    dnn.compile(optimizer=Adam(0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"])
    dnn.summary()
    dnn.fit(X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50, batch_size=1024, verbose=1,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
            ])
    dnn.save(os.path.join(MODELS_DIR, "behavioral_dnn.h5"))
    print("  Saved -> models/behavioral_dnn.h5")
    y_prob_dnn = dnn.predict(X_test, verbose=0).flatten()
    r_dnn = evaluate("DNN", y_test,
                     (y_prob_dnn >= 0.5).astype(int),
                     y_prob_dnn)
    logger.finish_model("DNN", {k: float(v) for k, v in r_dnn.items() if k in ("accuracy","precision","recall","f1","auc")})
    results.append(r_dnn)

    # ── Model 5: LSTM ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  MODEL 5: LSTM")
    print("="*55)
    logger.start_model("LSTM")
    X_tr_l = reshape_lstm(X_train)
    X_va_l = reshape_lstm(X_val)
    X_te_l = reshape_lstm(X_test)
    lstm = Sequential([
        LSTM(128, input_shape=(X_tr_l.shape[1], X_tr_l.shape[2]),
             return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(1, activation="sigmoid")
    ])
    lstm.compile(optimizer=Adam(0.001),
                 loss="binary_crossentropy",
                 metrics=["accuracy"])
    lstm.summary()
    lstm.fit(X_tr_l, y_train,
             validation_data=(X_va_l, y_val),
             epochs=50, batch_size=1024, verbose=1,
             callbacks=[
                 EarlyStopping(patience=5, restore_best_weights=True),
                 ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
             ])
    lstm.save(os.path.join(MODELS_DIR, "behavioral_lstm.h5"))
    print("  Saved -> models/behavioral_lstm.h5")
    y_prob_lstm = lstm.predict(X_te_l, verbose=0).flatten()
    r_lstm = evaluate("LSTM", y_test,
                      (y_prob_lstm >= 0.5).astype(int),
                      y_prob_lstm)
    logger.finish_model("LSTM", {k: float(v) for k, v in r_lstm.items() if k in ("accuracy","precision","recall","f1","auc")})
    results.append(r_lstm)

    model_bundle = {
        "Random Forest": rf,
        "XGBoost": xgb_m,
        "SVM": svm,
        "DNN": dnn,
        "LSTM": lstm,
    }
    logger.start_model("Meta-Ensemble")
    r_ens = train_meta_ensemble(model_bundle, X_val, y_val, X_test, y_test, external_eval=external_eval)
    logger.finish_model("Meta-Ensemble", {k: float(v) for k, v in r_ens.items() if k in ("accuracy","precision","recall","f1","auc")})
    results.append(r_ens)

    return results


def plot_results(results):
    colors = list(plt.cm.tab10(np.linspace(0, 1, max(len(results), 3))))

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("500K Dataset — Model Performance Comparison",
                 fontsize=13, fontweight="bold")
    for i, metric in enumerate(["accuracy", "precision", "recall", "f1", "auc"]):
        vals = [r[metric]*100 for r in results]
        bars = axes[i].bar([r["model"] for r in results], vals,
                           color=colors, edgecolor="white", width=0.5)
        axes[i].set_title(metric.capitalize(), fontsize=11)
        axes[i].set_ylim(80, 105)
        axes[i].tick_params(axis="x", rotation=30, labelsize=7)
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.3,
                         f"{val:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/500k_model_comparison.png",
                dpi=150, bbox_inches="tight")
    print(f"\n  Saved -> {FIGURES_DIR}/500k_model_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for r, color in zip(results, colors):
        if r.get("y_prob") is not None:
            fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{r['model']} (AUC={r['auc']*100:.2f}%)")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — 500K Dataset", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/500k_roc_curves.png",
                dpi=150, bbox_inches="tight")
    print(f"  Saved -> {FIGURES_DIR}/500k_roc_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    fig.suptitle("Confusion Matrices — 500K Dataset",
                 fontsize=13, fontweight="bold")
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(r["model"], fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/500k_confusion_matrices.png",
                dpi=150, bbox_inches="tight")
    print(f"  Saved -> {FIGURES_DIR}/500k_confusion_matrices.png")
    plt.close(fig)


def print_results(results):
    print("\n" + "="*65)
    print("   500K DATASET — FINAL MODEL COMPARISON")
    print("="*65)
    print(f"  {'Model':<18} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("  " + "-"*63)
    for r in results:
        print(f"  {r['model']:<18} {r['accuracy']*100:>8.2f}% "
              f"{r['precision']*100:>9.2f}% {r['recall']*100:>7.2f}% "
              f"{r['f1']*100:>7.2f}% {r['auc']*100:>7.2f}%")
    best = max(results, key=lambda x: x["f1"])
    print(f"\n  Best model: {best['model']}  (F1: {best['f1']*100:.2f}%)")
    print("="*65)
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ["y_pred", "y_prob", "y_true"]}
                       for r in results])
    df.to_csv("reports/500k_model_results.csv", index=False)
    print("  Saved -> reports/500k_model_results.csv")


def run_500k_training():
    print("=" * 55)
    print("  500K DATASET MODEL TRAINING")
    print("=" * 55)
    LOGGER.log("Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, feats, external_eval = load_data()
    LOGGER.log(f"Data loaded — Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")
    try:
        results = train_all(X_train, X_val, X_test, y_train, y_val, y_test, external_eval=external_eval, logger=LOGGER)
        print_results(results)
        plot_results(results)
        LOGGER.done()
        print("\n  Training complete!")
        print("  Run: streamlit run app/behavioral_dashboard.py")
    except Exception as exc:
        LOGGER.error(str(exc))
        raise
    return results


if __name__ == "__main__":
    run_500k_training()
