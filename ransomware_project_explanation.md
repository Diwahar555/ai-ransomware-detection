# AI-Based Ransomware Pre-Attack Detection System
### End-to-End Project Explanation

> **Project Type:** Final Year Cybersecurity / AI Project (2026)
> **Language:** Python
> **UI:** Streamlit Dashboard
> **Dataset:** 1,000,000 synthetic + real behavioral samples
> **Models:** Random Forest · XGBoost · SVM · DNN · LSTM + Meta-Ensemble

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Technology Stack & Dependencies](#3-technology-stack--dependencies)
4. [The Full Pipeline — Bird's Eye View](#4-the-full-pipeline--birds-eye-view)
5. [Stage 1 — Data Collection: Real-World Snapshots](#5-stage-1--data-collection-real-world-snapshots)
6. [Stage 2 — Dataset Generation (500K Synthetic Samples)](#6-stage-2--dataset-generation-500k-synthetic-samples)
7. [Stage 3 — Model Training (5 Models + Meta-Ensemble)](#7-stage-3--model-training-5-models--meta-ensemble)
8. [Stage 4 — Live Inference Engine](#8-stage-4--live-inference-engine)
9. [Stage 5 — False Positive Reduction](#9-stage-5--false-positive-reduction)
10. [Stage 6 — Anti-Evasion Engine](#10-stage-6--anti-evasion-engine)
11. [Stage 7 — Automated Response Engine](#11-stage-7--automated-response-engine)
12. [Stage 8 — Streamlit Dashboard](#12-stage-8--streamlit-dashboard)
13. [Stage 9 — Global Threat Archive](#13-stage-9--global-threat-archive)
14. [Reporting & Dashboard Utilities](#14-reporting--dashboard-utilities)
15. [File Activity Monitor](#15-file-activity-monitor)
16. [Real-World Data Capture](#16-real-world-data-capture)
17. [How All Modules Connect](#17-how-all-modules-connect)
18. [Key Design Decisions & Improvements](#18-key-design-decisions--improvements)
19. [How to Run the Project (Step-by-Step)](#19-how-to-run-the-project-step-by-step)
20. [Summary Table of Modules](#20-summary-table-of-modules)

---

## 1. Project Overview

This project is an **AI-powered ransomware pre-attack detection system** that monitors a live Windows machine in real time, analyzes behavioral features (CPU, memory, disk, network, file activity), and predicts whether ransomware is active — **before** files are encrypted.

### Core Idea

Traditional antivirus tools look for known file signatures. This project takes a completely different approach:

> **"Watch how the computer behaves, not what files it has."**

Ransomware has very distinct behavioral patterns:
- Sudden spike in CPU/memory usage
- Mass file modifications and deletions
- High disk write rates (encryption in progress)
- Unusual network connections (C&C server communication, data exfiltration)
- Many new processes spawning rapidly

The system uses **5 machine learning models** that vote on each behavioral snapshot. A **meta-ensemble** combines their votes for the final decision.

---

## 2. Project Structure

```
ransomware/
│
├── app/
│   └── behavioral_dashboard.py      ← Main Streamlit UI (1,500+ lines)
│
├── src/
│   ├── generate_500k_dataset.py     ← Dataset generator (1M samples)
│   ├── train_500k_models.py         ← Trains all 5 ML models + ensemble
│   ├── behavioral_predictor.py      ← Live real-time predictor
│   ├── anti_evasion.py              ← Anti-evasion detection engine
│   ├── false_positive_reducer.py    ← Multi-stage false positive filter
│   ├── enhanced_response_engine.py  ← Automated threat response actions
│   ├── detection_history.py         ← Thread-safe persistent threat archive
│   ├── report_generator.py          ← Intelligent A4 PDF incident generator
│   ├── dashboard_utils.py           ← Email alerts and heatmap generation
│   ├── file_activity_monitor.py     ← File system watcher (watchdog)
│   └── record_real_world_session.py ← Captures real labeled data
│
├── data/
│   └── behavioral/
│       ├── dataset_500k.csv         ← Generated dataset (~500 MB)
│       └── real_world/              ← Labeled real-world sessions
│
├── models/
│   ├── behavioral_rf.pkl            ← Random Forest model
│   ├── behavioral_xgb.pkl           ← XGBoost model
│   ├── behavioral_svm.pkl           ← Calibrated SVM model
│   ├── behavioral_dnn.h5            ← Deep Neural Network (Keras)
│   ├── behavioral_lstm.h5           ← LSTM model (Keras)
│   ├── behavioral_meta.pkl          ← Meta-ensemble (Logistic Regression)
│   ├── behavioral_scaler.pkl        ← StandardScaler
│   ├── behavioral_feature_names.pkl ← Feature name list
│   └── normal_stats.pkl             ← Normal behavior statistics
│
├── logs/
│   ├── live_data.csv                ← Live snapshot log
│   ├── incidents.log                ← Incident log
│   └── quarantine.json              ← Quarantined processes
│
├── reports/
│   ├── 500k_model_results.csv       ← Training metrics
│   ├── real_world_eval_results.csv  ← Real-world evaluation
│   └── figures/                     ← ROC curves, confusion matrices
│
├── tests/                           ← Unit tests
├── run_500k_dataset.py              ← Entry: generate dataset
├── run_500k_training.py             ← Entry: train models
├── run_real_world_capture.py        ← Entry: capture real data
└── requirements.txt
```

---

## 3. Technology Stack & Dependencies

| Library | Purpose |
|---|---|
| `psutil` | Live system metrics (CPU, memory, network, disk, processes) |
| `numpy` / `pandas` | Data processing and feature engineering |
| `scikit-learn` | Random Forest, SVM, StandardScaler, train/test split, metrics |
| `xgboost` | Gradient boosted trees model |
| `tensorflow` / `keras` | DNN and LSTM neural network models |
| `streamlit` | Live interactive web dashboard |
| `plotly` | Interactive charts in the dashboard |
| `matplotlib` | Static training plots and heatmaps |
| `reportlab` | PDF incident report generation |
| `watchdog` | File system event monitoring |
| `plyer` | Windows desktop notification popups |

---

## 4. The Full Pipeline — Bird's Eye View

```
STEP 1: Collect 1,000 real normal snapshots (~8 min)
        (while using the laptop normally)
        |
        v
STEP 2: Generate 1,000,000-sample synthetic dataset
   - 500,000 normal samples (real + synthesized)
   - 500,000 attack samples (5 ransomware variants)
   - Add overlap/noise for realism
        |
        v
STEP 3: Train 5 ML models + meta-ensemble
   Random Forest -> XGBoost -> SVM -> DNN -> LSTM
                     vote probabilities
              Meta-Ensemble (LogReg)
   Saves: .pkl / .h5 / scaler / feature names
        |
        v
STEP 4: Run Live Dashboard
   Every 6 seconds:
   1. Collect live behavioral snapshot (psutil)
   2. Predict with 5 models -> ensemble confidence
   3. False Positive Reducer (context + multi-stage)
   4. Anti-Evasion Engine (sliding window, entropy)
   5. If threat confirmed -> Response Engine activates
   6. Dashboard shows live charts, alerts, actions
```

---

## 5. Stage 1 — Data Collection: Real-World Snapshots

**File:** `src/generate_500k_dataset.py` → `collect_real_normal()`

The first step gathers **1,000 real behavioral snapshots** from the user's own machine over approximately 8 minutes. This "seeds" the entire dataset with realistic normal behavior.

### What is collected every 0.5 seconds?

| Feature | Source | Description |
|---|---|---|
| `cpu_percent` | `psutil.cpu_percent()` | Overall CPU usage percentage |
| `memory_percent` | `psutil.virtual_memory().percent` | RAM usage percentage |
| `process_count` | `psutil.process_iter()` | Total number of running processes |
| `high_cpu_process_count` | Processes with CPU > 5% | Count of CPU-heavy processes |
| `active_connections` | `psutil.net_connections()` | Total network connections |
| `established_connections` | Status == ESTABLISHED | Active TCP connections |
| `unique_remote_ports` | `c.raddr.port` | Unique destination ports |
| `bytes_sent_rate` | Delta bytes_sent / delta time | Network upload speed (bytes/sec) |
| `bytes_recv_rate` | Delta bytes_recv / delta time | Network download speed |
| `file_modified_count` | File watcher events | File modifications in interval |
| `file_created_count` | File watcher events | File creations in interval |
| `file_deleted_count` | File watcher events | File deletions in interval |
| `disk_write_rate` | Delta write_bytes / delta time | Disk write throughput (bytes/sec) |
| `new_process_count` | New PIDs vs previous snapshot | Newly started processes |

These **14 features** form the input vector for all ML models.

---

## 6. Stage 2 — Dataset Generation (500K Synthetic Samples)

**File:** `src/generate_500k_dataset.py` → `build_1m_dataset()`

### Why synthetic data?

You cannot collect 1,000,000 real ransomware attack samples safely. Instead, the system:
1. Studies the statistical distribution of real normal behavior
2. Synthesizes additional normal data preserving correlations
3. Generates attack data by modeling how ransomware changes these features

### Normal Sample Generation

```
Real 1,000 snapshots
         |
Compute covariance matrix (feature correlations)
         |
Bootstrap sampling (reuse real rows) + multivariate normal drawing
         |
Blend both: weighted average (35-75% bootstrap, rest correlated)
         |
Add measurement noise (8-10% of std)
         |
Inject benign bursts (8% of samples get brief harmless spikes)
         |
Clip/validate all constraint rules (connections >= established)
         |
499,000 synthetic normal samples
```

### Attack Sample Generation — 5 Ransomware Variants

| Variant | Based On | Key Behavior |
|---|---|---|
| **Fast Encryption** | WannaCry-style | Very high CPU (55-99%), huge disk writes (2-100 MB/s), mass file mods/deletes |
| **Slow Encryption** | Evasion-focused | Low-and-slow (25-68% CPU), modest file ops — designed to evade detection |
| **Fileless Attack** | Memory-resident | High memory (65-99%), heavy network (50KB-1MB/s), few disk writes |
| **Polymorphic** | Phase-changing | Cycles through 4 phases: encryption, exfiltration, process spawn, quiet |
| **Network-Heavy** | Ryuk/REvil-style | Mass connections (25-200), high upload (80KB-5MB/s), moderate encryption |

### Severity Vector

Each attack sample gets a random severity score (0.25 to 1.0) drawn from a Beta distribution. A random `overlap_ratio` (18-35%) of attack samples are blended back toward normal behavior — making the detection task realistically hard.

### Class Overlap Injection

- **8% of normal samples** get borderline attack-like characteristics
- **18% of attack samples** get blended toward normal
- This prevents models from learning unrealistically clean boundaries

### Final Dataset

```
Total samples : 1,000,000
Normal  (0)   : 500,000
Attack  (1)   : 500,000
Features      : 14
```

Saved to: `data/behavioral/dataset_500k.csv`

---

## 7. Stage 3 — Model Training (5 Models + Meta-Ensemble)

**File:** `src/train_500k_models.py`

### Data Split

```
1,000,000 samples
    |-- 70% Training   (700,000)
    |-- 15% Validation (150,000)  <- used for meta-ensemble training
    |-- 15% Test       (150,000)  <- final evaluation
```

A `StandardScaler` is fitted on training data and saved as `behavioral_scaler.pkl`.

### Model 1 — Random Forest

- 200 decision trees, max depth 20, class balanced
- Saved as `behavioral_rf.pkl`
- **Why:** Robust, handles feature interactions, gives probability outputs. Ensemble of trees reduces variance.

### Model 2 — XGBoost

- 300 boosted trees, learning rate 0.05, histogram method
- Saved as `behavioral_xgb.pkl`
- **Why:** Often the best performer on tabular data. Uses boosting — each tree corrects errors of previous ones.

### Model 3 — SVM (Calibrated)

- LinearSVC wrapped with CalibratedClassifierCV(cv=3, method="sigmoid")
- Trained on 30,000 samples (not scalable to 700K)
- Saved as `behavioral_svm.pkl`
- **Why:** Good at linear decision boundaries; CalibratedClassifierCV gives probabilities.

### Model 4 — Deep Neural Network (DNN)

```
Dense(256, relu) -> BatchNorm -> Dropout(0.3)
Dense(128, relu) -> BatchNorm -> Dropout(0.3)
Dense(64,  relu) -> BatchNorm -> Dropout(0.2)
Dense(32,  relu)
Dense(1, sigmoid)
```
- Adam optimizer, binary cross-entropy loss, EarlyStopping
- Saved as `behavioral_dnn.h5`

### Model 5 — LSTM

```
LSTM(128, return_sequences=True) -> Dropout(0.3)
LSTM(64)  -> Dropout(0.2)
Dense(32, relu) -> BatchNorm
Dense(1, sigmoid)
```
- Input reshaped to (samples, 2 timesteps, 7 features)
- Saved as `behavioral_lstm.h5`

### Meta-Ensemble (Stacking)

```
All 5 models predict probabilities on VALIDATION set
          |
Stack probabilities as 5 features per sample
          |
Train LogisticRegression meta-learner on stacked outputs
          |
Final probability = meta-learner output
```
- Saved as `behavioral_meta.pkl`
- **Fallback weighted average:** RF=25%, XGBoost=30%, SVM=15%, DNN=15%, LSTM=15%

### Training Reports

- `reports/500k_model_results.csv` — per-model accuracy, precision, recall, F1, AUC
- `reports/figures/500k_model_comparison.png` — bar chart comparison
- `reports/figures/500k_roc_curves.png` — ROC curves
- `reports/figures/500k_confusion_matrices.png` — confusion matrices

---

## 8. Stage 4 — Live Inference Engine

**File:** `src/behavioral_predictor.py`

### Loading Models

At startup, `load_behavioral_models()` loads all model files and the scaler.

### Prediction Flow (`predict_behavioral`)

```
snapshot dict (14 features)
         |
Convert to DataFrame, fill NaN, clip infinities
         |
scaler.transform() -- normalize to training distribution
         |
All 5 models predict probability of attack
         |
Vote: is_threat = (prob >= threshold) True/False per model
         |
Meta-ensemble combines 5 probabilities -> final confidence score
         |
Return: {is_threat, confidence, vote_count, votes, probabilities}
```

### Output Example

```json
{
  "is_threat": false,
  "confidence": 0.2341,
  "vote_count": 1,
  "votes": {"Random Forest": 0, "XGBoost": 0, "SVM": 1, "DNN": 0, "LSTM": 0},
  "probabilities": {"Random Forest": 0.12, "XGBoost": 0.18, ...},
  "ensemble_method": "calibrated_meta"
}
```

---

## 9. Stage 5 — False Positive Reduction

**File:** `src/false_positive_reducer.py`

Normal computers regularly spike in CPU, memory, and disk — during antivirus scans, Windows Update, backups, video rendering, etc. Without FP reduction, the system would generate constant false alarms.

### The FalsePositiveReducer Pipeline (5 Steps)

**Step 1 — Confidence Smoothing**

Exponential Moving Average: `smoothed = 0.4 × raw + 0.6 × previous_smoothed`

Prevents a single-spike snapshot from triggering an alert.

**Step 2 — Context Analysis**

Applies penalties to reduce confidence when behavior looks benign:

| Context Check | Penalty |
|---|---|
| No file activity + low disk write | −0.35 |
| Very low file activity | −0.20 |
| High CPU but NO untrusted processes | −0.15 |
| Disk spike during scheduled backup hours (0-3 AM) | −0.10 |
| High CPU at Windows Update hour (2-4 AM) | −0.10 |
| No suspicious process activity | −0.10 |
| Normal network (no burst) | −0.05 |

**Step 3 — Time-Adjusted Threshold**

Late night (00:00-06:00) uses a lower threshold (more sensitive).  
Business hours (09:00-18:00) uses the base threshold.

**Step 4 — Evidence Gate**

Even if confidence is high, a threat is only flagged when suspicion_score >= 0.20 (supporting behavioral evidence exists).

**Step 5 — Multi-Stage Confirmation**

Requires **3 consecutive** threat detections (18 seconds) before confirming. A single bad reading never triggers response.

```
Check 1: Threat? -> stage 1/3
Check 2: Threat? -> stage 2/3
Check 3: Threat? -> CONFIRMED!
Check 4: Normal? -> reset to 0/3
```

---

## 10. Stage 6 — Anti-Evasion Engine

**File:** `src/anti_evasion.py`

Sophisticated ransomware tries to evade detection by encrypting slowly. The engine watches for slow-burn patterns.

**SlidingWindowAnalyzer (60-second window):**
- Cumulative file modifications > 100: +0.35 weight
- Sustained CPU > 70% for 60%+ of window: +0.25 weight
- Disk write doubling AND > 5 MB/s: +0.20 weight
- Cumulative deletions > 50: +0.20 weight
- Evasion detected if total >= 0.50

**BehavioralDriftDetector:**
- Builds baseline from first 20 snapshots
- Detects drift using z-scores (flags if z > 4.0 standard deviations)

**EntropySpikeDetector:**
- Scores per-snapshot encryption activity (disk writes, file changes, CPU, new processes)
- Spike if score > 0.60, sustained if rolling average > 0.50

**EvidenceAccumulator:**
- Decays 30% each check (evidence = evidence × 0.70)
- Only grows when BOTH sliding_score > 0.2 AND entropy_score > 0.2
- Resets rapidly if model confidence stays below 30% for 3+ checks
- Alert when accumulated evidence >= 0.75

**Final Rule:** Evasion is flagged only when at least TWO independent signals agree.

---

## 11. Stage 7 — Automated Response Engine

**File:** `src/enhanced_response_engine.py`

When a threat is confirmed (all toggles are opt-in from the dashboard):

**1. Network Isolation**
```
netsh advfirewall firewall add rule name="RansomwareDetectionBlock" 
      dir=out action=block protocol=any
```
Blocks outbound traffic to stop C&C communication and data exfiltration.

**2. File Protection**
```
icacls ~/Documents /deny Everyone:(W,D,DC) /T
```
ACL-denies write/delete on Desktop, Documents, Pictures, Downloads.

**3. Process Quarantine/Termination**

Risk scoring per running process:

| Factor | Points |
|---|---|
| Name contains "encrypt", "ransom", "cipher" etc. | +5 |
| Running from user-writable path (Temp, Roaming) | +4 |
| Spawned by PowerShell/cmd/wscript | +2 |
| Suspicious command-line args (vssadmin, .locked) | +3 |
| CPU >= 70% | +5 |
| Running from trusted install path | −3 |

- Score >= 6 → flagged for review
- Score >= 8 → automatically suspended or killed

**4. Desktop Notification** — Windows toast popup via `plyer` or PowerShell fallback.

**5. Incident Logging** — Written to `logs/incidents.log`.

---

## 12. Stage 8 — Streamlit Dashboard

**File:** `app/behavioral_dashboard.py`

Live web dashboard at `http://localhost:8501`. Auto-refreshes every 6 seconds.
The dashboard features an ultra-modern, professional light-mode aesthetic heavily inspired by enterprise Security Operations Centers (SOC).

### Sections
- **Live Overview Tab:** Real-time metrics, rolling timeseries charts, anti-evasion telemetry, model voting panel.
- **Operations Tab:** Detailed engine configuration overrides, prevention status, error handling details.
- **Model Intelligence Tab:** Performance metrics on raw datasets vs. held-out attack datasets.
- **Archive & History Tab:** Interactive dataframe directly polling `detection_history.py` — users can view, filter, resolve, and generate reports for threats historically confirmed by the system.
- **SOC Side Panel:** Manual adjustments, threat sensitivity knobs, automated response toggles, and email alert integration.

---

## 13. Stage 9 — Global Threat Archive

**File:** `src/detection_history.py`

A critical component to preserve the history of threats actively blocked across system deployments/server reboots.
- Built using a **Thread-Safe singleton** architecture.
- Serializes real-time mitigation data into a secure fallback `logs/detection_history.json`.
- Maintains the specific processes halted, model scoring at time of quarantine, and automated containment actions executed (like folder locking or network quarantine).

---

## 14. Reporting & Dashboard Utilities

**Files:** `src/report_generator.py` and `src/dashboard_utils.py`

- **Intelligent PDF Reporting:** Through the `report_generator.py` using **ReportLab**, analysts can click standard endpoints to dynamically pull the global archive dataset, formatting it beautifully into A4 PDFs detailing total session analytics, model configurations, and threat actions.
- **Email Alerts:** HTML email with red header, metric table, recommended actions. Password never persisted to disk.
- **Threat Heatmap:** Dual bar charts — average confidence and threat count by hour of day.

---

## 14. File Activity Monitor

**File:** `src/file_activity_monitor.py`

- Uses `watchdog` library (or polling fallback) to detect file system events
- Watches: Desktop, Documents, Downloads, Pictures
- Counts modified/created/deleted events per 6-second interval
- Thread-safe counter fed directly into live snapshot collection

---

## 15. Real-World Data Capture

**File:** `src/record_real_world_session.py`

```powershell
# Normal session (label=0)
python run_real_world_capture.py --label 0 --duration 300 --session-id normal_office_01

# Attack simulation (label=1)
python run_real_world_capture.py --label 1 --duration 180 --session-id attack_sim_01
```

Saves CSVs to `data/behavioral/real_world/`. The training script automatically detects these, splits by session (to prevent leakage), and evaluates separately in `real_world_eval_results.csv`.

---

## 16. How All Modules Connect

```
behavioral_dashboard.py (Streamlit Main)
   |
   |-- behavioral_predictor.py      <- loads models, collects snapshots, runs prediction
   |       |-- models/*.pkl, *.h5   <- pre-trained ML models
   |       |-- file_activity_monitor.py <- file event counts
   |
   |-- false_positive_reducer.py    <- smoothing, context, confirmation
   |
   |-- anti_evasion.py              <- sliding window, drift, entropy, evidence
   |
   |-- enhanced_response_engine.py  <- network, ACL, process quarantine
   |
   |-- dashboard_utils.py           <- email, PDF, heatmap
```

### Each 6-Second Tick Flow

```
1. file_activity_monitor -> {modified, created, deleted}
2. collect_live_snapshot(file_events) -> 14-feature snapshot dict
3. predict_behavioral(snapshot) -> raw_result (5 model probs + ensemble)
4. false_positive_reducer.process(raw_result, snapshot) -> fp_result
5. anti_evasion.analyze(snapshot, confidence) -> ae_result
6. if fp_result["confirmed"] AND toggles enabled:
        respond_to_threat(...)
7. All results displayed on dashboard
8. If email configured + threat confirmed -> send_email_alert()
```

---

## 17. Key Design Decisions & Improvements

**Why 5 Models?**
No single model is perfect. Tree models catch feature patterns, SVM catches linear boundaries, DNN catches non-linear interactions, LSTM adds temporal awareness. Meta-ensemble learns optimal combination weights.

**Why Synthetic Data?**
Real ransomware is dangerous — can't run 500,000 attack sessions safely. Synthetic data generated from real distributions is statistically valid and covers 5 major ransomware behavioral archetypes.

**False Positive Reduction is Critical**
A system that cries wolf every few minutes becomes useless. Multi-stage pipeline ensures alerts are meaningful and actionable.

**Anti-Evasion Addresses Real Attacks**
Sophisticated ransomware (Sodinokibi/REvil) throttles CPU to evade detection. Sliding window and entropy detectors catch patterns that per-snapshot models miss.

**Safe Process Whitelist**
The response engine won't kill browsers, VS Code, Python, or system processes — even if high CPU. Prevents the cure from being worse than the disease.

**Email Password Never Stored**
If this machine is compromised, the attacker should not find credentials. Email passwords are kept only in session memory.

---

## 18. How to Run the Project (Step-by-Step)

### Install dependencies

```powershell
cd "v:\New folder (3)\ransomeware"
pip install -r requirements.txt
```

### Step 1 — Generate the Dataset (~10 minutes)

```powershell
python run_500k_dataset.py
```
- Collects 1,000 real snapshots while you use the laptop normally
- Generates 999,000 synthetic samples
- Saves `data/behavioral/dataset_500k.csv`

### Step 2 — Train the Models (~30-60 minutes)

```powershell
python run_500k_training.py
```
- Trains RF, XGBoost, SVM, DNN, LSTM + meta-ensemble
- Saves 6 model files to `models/`
- Generates evaluation charts in `reports/figures/`

### Step 3 (Optional) — Capture Real-World Data

```powershell
python run_real_world_capture.py --label 0 --duration 300 --session-id normal_01
```

### Step 4 — Launch the Dashboard

```powershell
streamlit run app/behavioral_dashboard.py
```
Open browser: `http://localhost:8501`

### Step 5 — Run Tests

```powershell
python -m unittest discover -s tests -v
```

---

## 20. Summary Table of Modules

| File | Role | Key Functions |
|---|---|---|
| `generate_500k_dataset.py` | Dataset generation | `collect_real_normal`, `build_1m_dataset`, `gen_fast_encryption`, `...` |
| `train_500k_models.py` | Model training | `load_data`, `train_all`, `train_meta_ensemble`, `plot_results` |
| `behavioral_predictor.py` | Live inference | `load_behavioral_models`, `collect_live_snapshot`, `predict_behavioral` |
| `false_positive_reducer.py` | FP reduction | `FalsePositiveReducer`, `ConfidenceSmoother`, `MultiStageConfirmation` |
| `anti_evasion.py` | Evasion detection | `AntiEvasionEngine`, `SlidingWindowAnalyzer`, `BehavioralDriftDetector` |
| `enhanced_response_engine.py` | Automated response | `respond_to_threat`, `isolate_network`, `protect_files`, `quarantine_process` |
| `detection_history.py` | Persistent logging | `DetectionHistory` (thread-safe, JSON-backed logs) |
| `report_generator.py` | PDF exporting | `generate_threat_report` |
| `dashboard_utils.py` | Dashboard extras | `send_email_alert`, `generate_heatmap` |
| `file_activity_monitor.py` | File watching | `FileActivityMonitor` (watchdog-based) |
| `record_real_world_session.py` | Data labeling | `record_session` |
| `behavioral_dashboard.py` | Streamlit UI | Premium active monitoring, SOC dashboard frontend |

---

> **Document prepared for:** Final Year Project Submission 2026
> **System:** AI-Based Ransomware Pre-Attack Detection System
> **Coverage:** End-to-end — from data collection through live deployment
