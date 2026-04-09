# Ransomware Security Operations Dashboard - Setup Guide

Follow these steps to set up and run the AI-Based Ransomware Pre-Attack Prediction System on a new Windows machine.

## 1. Prerequisites
- **Python 3.10+**: Ensure Python is installed. You can check with `python --version`.
- **Git**: (Optional) To clone the repository.

## 2. Initial Setup
Open a terminal (PowerShell or Command Prompt) in the project root directory.

### Create a Virtual Environment
```powershell
# Create the environment
python -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1

# OR Activate it (CMD)
.\venv\Scripts\activate.bat
```

### Install Dependencies
```powershell
pip install -r requirements.txt
```

## 3. Training the Models
Before starting the dashboard, you must generate the synthetic training data and train the AI models.

```powershell
# Step 1: Generate the behavioral dataset
python run_500k_dataset.py

# Step 2: Train the ensemble model (RF, XGB, SVM, DNN, LSTM)
python run_500k_training.py
```
*Note: This might take several minutes depending on your hardware.*

## 4. Launching the Dashboard
Once the models are trained, you can start the monitoring console:

```powershell
streamlit run app/behavioral_dashboard.py
```
The dashboard will open automatically in your browser (usually at `http://localhost:8501`).

## 5. Security Testing (Attack Simulator)
To test the detection engine, you can run the provided attack simulator in a separate terminal:

```powershell
# Ensure venv is activated
python attack_simulator.py
```
*Caution: The simulator mimics ransomware behavior for detection testing. Run it in a controlled environment.*

## 6. Troubleshooting
- **Model not found**: Ensure you ran the training scripts in step 3.
- **Permission Denied**: Run the terminal as Administrator if the system prevents `psutil` from reading process telemetry.
- **Port 8501 in use**: Streamlit will offer to run on the next available port (e.g., 8502).

---
**Project Structure:**
- `/app`: Streamlit UI files
- `/src`: Core logic (Anti-Evasion, Response Engine, Predictors)
- `/models`: Saved AI model artifacts (.pkl, .h5)
- `/reports`: Generated performance metrics
