# Ransomware Detection Project

This project trains an offline behavioral ensemble and uses it in a Streamlit dashboard for live ransomware-risk monitoring, anti-evasion checks, and optional response actions.

## Pipeline

`real snapshots -> synthetic dataset -> training -> saved scaler/models -> live inference`

Additional labeled host data can now be captured into `data/behavioral/real_world/` and folded into training plus external evaluation.

## Main commands

Generate the synthetic dataset:

```powershell
python run_500k_dataset.py
```

Train the models and calibrated ensemble:

```powershell
python run_500k_training.py
```

Capture labeled real-world data:

```powershell
python run_real_world_capture.py --label 0 --duration 300 --interval 3 --session-id normal_office_01
python run_real_world_capture.py --label 1 --duration 180 --interval 2 --session-id attack_sim_01
```

Run the dashboard:

```powershell
streamlit run app/behavioral_dashboard.py
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

## Notes

- Email passwords are session-only and are not stored on disk.
- File activity monitoring uses `watchdog` when available and falls back to polling otherwise.
- `reports/500k_model_results.csv` contains synthetic holdout metrics.
- `reports/real_world_eval_results.csv` is generated only when labeled real-world sessions exist.
- Response actions can change firewall rules, process state, and folder ACLs, so review dashboard toggles carefully.
