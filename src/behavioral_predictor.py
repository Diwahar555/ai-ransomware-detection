"""
behavioral_predictor.py
Real-time predictor using behavioral features.
"""

import os
import csv
import pickle
import numpy as np
import pandas as pd
import datetime
import psutil
import time
import logging
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

MODELS_DIR = "models"
MODEL_ORDER = ["Random Forest", "XGBoost", "SVM", "DNN", "LSTM"]

FEATURE_NAMES = [
    "cpu_percent",
    "memory_percent",
    "process_count",
    "high_cpu_process_count",
    "active_connections",
    "established_connections",
    "unique_remote_ports",
    "bytes_sent_rate",
    "bytes_recv_rate",
    "file_modified_count",
    "file_created_count",
    "file_deleted_count",
    "disk_write_rate",
    "new_process_count",
]

_state = {"net": None, "disk": None, "time": None, "pids": set()}


def load_behavioral_models():
    print("-- Loading behavioral models -------------------------")
    with open(os.path.join(MODELS_DIR, "behavioral_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    print("  behavioral_scaler.pkl  loaded")

    models = {}
    for name, fname in [("Random Forest", "behavioral_rf.pkl"),
                         ("XGBoost",       "behavioral_xgb.pkl"),
                         ("SVM",           "behavioral_svm.pkl")]:
        with open(os.path.join(MODELS_DIR, fname), "rb") as f:
            models[name] = pickle.load(f)
        if name == "Random Forest" and hasattr(models[name], "n_jobs"):
            # Threaded RF inference can hit permission issues on some Windows setups.
            models[name].n_jobs = 1
            if hasattr(models[name], "verbose"):
                models[name].verbose = 0
        print(f"  {fname:25s} loaded")

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    for name, fname in [("DNN",  "behavioral_dnn.h5"),
                         ("LSTM", "behavioral_lstm.h5")]:
        models[name] = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, fname), compile=False)
        print(f"  {fname:25s} loaded")

    ensemble_model = None
    meta_path = os.path.join(MODELS_DIR, "behavioral_meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            ensemble_model = pickle.load(f)
        print("  behavioral_meta.pkl     loaded")
    else:
        print("  behavioral_meta.pkl     not found (using fallback weights)")

    print("  Behavioral models ready.\n")
    return models, scaler, ensemble_model


def collect_live_snapshot(file_events=None):
    global _state
    now  = time.time()
    snap = {}

    snap["cpu_percent"]    = psutil.cpu_percent(interval=0.2)
    snap["memory_percent"] = psutil.virtual_memory().percent

    try:
        procs = list(psutil.process_iter(["pid", "cpu_percent", "status"]))
        current_pids = set(p.info["pid"] for p in procs)
        snap["process_count"]          = len(procs)
        snap["high_cpu_process_count"] = sum(1 for p in procs
                                             if (p.info.get("cpu_percent") or 0) > 15)
        # Guard: if _state["pids"] is empty this is the very first call;
        # report 0 new processes instead of ALL processes on the system.
        if _state["pids"]:
            snap["new_process_count"] = len(current_pids - _state["pids"])
        else:
            snap["new_process_count"] = 0
        _state["pids"] = current_pids
    except Exception:
        snap["process_count"]          = 100
        snap["high_cpu_process_count"] = 2
        snap["new_process_count"]      = 0

    try:
        net   = psutil.net_io_counters()
        conns = psutil.net_connections()
        snap["active_connections"]      = len(conns)
        snap["established_connections"] = sum(1 for c in conns if c.status == "ESTABLISHED")
        snap["unique_remote_ports"]     = len(set(c.raddr.port for c in conns if c.raddr))
        if _state["net"] and _state["time"]:
            dt = max(now - _state["time"], 0.1)
            snap["bytes_sent_rate"] = max(0, (net.bytes_sent - _state["net"].bytes_sent) / dt)
            snap["bytes_recv_rate"] = max(0, (net.bytes_recv - _state["net"].bytes_recv) / dt)
        else:
            snap["bytes_sent_rate"] = 0.0
            snap["bytes_recv_rate"] = 0.0
        _state["net"] = net
    except Exception:
        snap["active_connections"]      = 30
        snap["established_connections"] = 10
        snap["unique_remote_ports"]     = 5
        snap["bytes_sent_rate"]         = 0.0
        snap["bytes_recv_rate"]         = 0.0

    try:
        disk = psutil.disk_io_counters()
        if _state["disk"] and _state["time"]:
            dt = max(now - _state["time"], 0.1)
            snap["disk_write_rate"] = max(0, (disk.write_bytes - _state["disk"].write_bytes) / dt)
        else:
            snap["disk_write_rate"] = 0.0
        _state["disk"] = disk
    except Exception:
        snap["disk_write_rate"] = 0.0

    if file_events:
        snap["file_modified_count"] = file_events.get("modified", 0)
        snap["file_created_count"]  = file_events.get("created",  0)
        snap["file_deleted_count"]  = file_events.get("deleted",  0)
    else:
        snap["file_modified_count"] = 0
        snap["file_created_count"]  = 0
        snap["file_deleted_count"]  = 0

    _state["time"] = now
    snap["timestamp"] = datetime.datetime.now().isoformat()

    # Save to CSV log
    os.makedirs("logs", exist_ok=True)
    log_path    = "logs/live_data.csv"
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_NAMES + ["timestamp"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: snap.get(k, 0) for k in FEATURE_NAMES + ["timestamp"]})

    return snap


def _reshape_lstm(X, timesteps=2):
    n     = X.shape[1]
    n_per = n // timesteps
    pad   = (timesteps * n_per) - n
    if pad < 0:
        n_per += 1
        pad = (timesteps * n_per) - n
    if pad > 0:
        X = np.pad(X, ((0, 0), (0, pad)))
    return X.reshape(X.shape[0], timesteps, n_per)


def _collect_model_probabilities(X, models):
    probs = {}

    for name in ["Random Forest", "XGBoost", "SVM"]:
        probs[name] = float(models[name].predict_proba(X)[0][1])

    dnn_p = float(models["DNN"].predict(X, verbose=0).flatten()[0])
    probs["DNN"] = dnn_p

    lstm_status = "Active"
    try:
        # Validate input shape: (samples, timesteps, features)
        X_l = _reshape_lstm(X.copy(), timesteps=2)
        if X_l.shape != (1, 2, 7): # Expected shape for this model
            raise ValueError(f"Invalid LSTM input shape: {X_l.shape}, expected (1, 2, 7)")
            
        prediction = models["LSTM"].predict(X_l, verbose=0).flatten()[0]
        
        # Check for zero-prediction or NaN (usually indicating broken weights/input)
        if np.isnan(prediction) or prediction == 0.0:
            raise ValueError("LSTM returned invalid or zero prediction")
            
        probs["LSTM"] = float(prediction)
    except Exception as e:
        probs["LSTM"] = float(dnn_p * 0.9)
        lstm_status = "Fallback Mode"
        
        # Log the fallback event
        os.makedirs("logs", exist_ok=True)
        with open("logs/incidents.log", "a") as f:
            f.write(f"\n[{datetime.datetime.now().isoformat()}] LSTM STABILITY FALLBACK: {str(e)}\n")
        print(f"  [!] LSTM Fallback engaged: {e}")

    probs["lstm_status"] = lstm_status

    # Neural networks trained on synthetically boosted tabular data are prone to 
    # violent overconfidence (outputting exactly 1.0 even on normal noise).
    # If the tree models (which generalize better) uniformly agree the threat is low, 
    # we penalize the DNN/LSTM hallucination mathematically.
    tree_avg = (probs["Random Forest"] + probs["XGBoost"]) / 2.0
    if tree_avg < 0.3:
        # Pull the networks back toward reality
        probs["DNN"] = probs["DNN"] * 0.2 + tree_avg * 0.8
        probs["LSTM"] = probs["LSTM"] * 0.2 + tree_avg * 0.8

    return probs


def _ensemble_probability(probs, ensemble_model=None):
    if ensemble_model is not None and hasattr(ensemble_model, "predict_proba"):
        meta_features = np.array([[probs[name] for name in MODEL_ORDER]], dtype=np.float32)
        return float(ensemble_model.predict_proba(meta_features)[0][1]), "calibrated_meta"

    weights = {"Random Forest": 0.25, "XGBoost": 0.30,
               "SVM": 0.15, "DNN": 0.15, "LSTM": 0.15}
    return sum(probs[name] * weights[name] for name in MODEL_ORDER), "weighted_fallback"


def classify_attack_type(snapshot, ae_result=None):
    """
    Update 4: Classify ransomware type based on behavioral features.
    """
    cpu = snapshot.get("cpu_percent", 0)
    disk_rate = snapshot.get("disk_write_rate", 0) / 1e6 # MB/s
    mem = snapshot.get("memory_percent", 0)
    net_sent = snapshot.get("bytes_sent_rate", 0) / 1e6 # MB/s
    conns = snapshot.get("active_connections", 0)
    files_mod = snapshot.get("file_modified_count", 0)
    
    # Extract scores from AE engine if available
    entropy = 0
    drift = 0
    if ae_result:
        entropy = ae_result.get("entropy_analysis", {}).get("score", 0)
        drift = ae_result.get("behavioral_drift", {}).get("score", 0)

    # 1. WannaCry-style (Fast)
    if cpu > 80 and disk_rate > 100:
        return {
            "type": "WannaCry-style (Fast)",
            "desc": "High-velocity encryption targeting maximum file damage in minimum time."
        }
    
    # 2. Network-Heavy (Ryuk/REvil)
    if net_sent > 50 and conns > 100:
        return {
            "type": "Network-Heavy (Ryuk/REvil)",
            "desc": "Focuses on data exfiltration and lateral movement across the network."
        }
    
    # 3. Slow/Evasive
    if disk_rate < 10 and entropy > 0.6:
        return {
            "type": "Slow/Evasive",
            "desc": "Low-noise encryption designed to evade behavioral triggers by throttling activity."
        }
    
    # 4. Fileless
    if mem > 70 and disk_rate < 2 and conns > 20:
        return {
            "type": "Fileless",
            "desc": "Operates entirely in memory, avoiding disk-based detection signatures."
        }
        
    # 5. Polymorphic
    if drift > 0.7:
        return {
            "type": "Polymorphic",
            "desc": "Rapidly changes its behavior and process profile to bypass static pattern matching."
        }
        
    return {
        "type": "Generalized Ransomware",
        "desc": "Standard behavioral patterns including unusual resource spikes and file access."
    }


def predict_behavioral(snapshot, models, scaler, threshold=0.5, ensemble_model=None):
    row = {feat: snapshot.get(feat, 0) for feat in FEATURE_NAMES}
    df  = pd.DataFrame([row])[FEATURE_NAMES].fillna(0)
    df  = df.replace([np.inf, -np.inf], 0)
    X   = scaler.transform(df.values.astype(np.float32))

    probs = _collect_model_probabilities(X, models)
    votes = {name: int(probs[name] >= threshold) for name in MODEL_ORDER}
    confidence, ensemble_method = _ensemble_probability(probs, ensemble_model)
    is_threat  = confidence >= threshold

    return {
        "is_threat":     is_threat,
        "confidence":    round(confidence, 4),
        "vote_count":    sum(votes.values()),
        "votes":         votes,
        "probabilities": probs,
        "lstm_status":   probs.get("lstm_status", "Active"),
        "ensemble_method": ensemble_method,
        "timestamp":     datetime.datetime.now().isoformat(),
        "snapshot":      snapshot,
    }
