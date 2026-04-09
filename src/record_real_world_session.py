"""
record_real_world_session.py
Capture labeled live behavioral data for real-world training/evaluation.
"""

import argparse
import os
import socket
import time

import pandas as pd

from src.behavioral_predictor import FEATURE_NAMES, collect_live_snapshot
from src.file_activity_monitor import FileActivityMonitor, get_default_watch_paths


REAL_WORLD_DIR = os.path.join("data", "behavioral", "real_world")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture a labeled behavioral session.")
    parser.add_argument("--label", type=int, choices=[0, 1], required=True,
                        help="0 for benign activity, 1 for attack/simulation.")
    parser.add_argument("--duration", type=int, default=300,
                        help="Capture duration in seconds.")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Seconds between snapshots.")
    parser.add_argument("--session-id", type=str, required=True,
                        help="Unique session identifier, e.g. normal_office_01.")
    parser.add_argument("--host-id", type=str, default=socket.gethostname(),
                        help="Host identifier stored alongside the session.")
    parser.add_argument("--notes", type=str, default="",
                        help="Optional free-form session notes.")
    return parser.parse_args()


def capture_session(label, duration, interval, session_id, host_id, notes=""):
    os.makedirs(REAL_WORLD_DIR, exist_ok=True)
    output_path = os.path.join(REAL_WORLD_DIR, f"{session_id}.csv")
    monitor = FileActivityMonitor(get_default_watch_paths()).start()

    rows = []
    total_steps = max(int(duration / max(interval, 0.5)), 1)
    print(f"-- Capturing session '{session_id}' for ~{duration}s ({total_steps} samples) --")
    print(f"   Host: {host_id} | Label: {label} | Output: {output_path}")

    try:
        for step in range(total_steps):
            file_events = monitor.get_counts(reset=True)
            snapshot = collect_live_snapshot(file_events=file_events)
            row = {feature: snapshot.get(feature, 0) for feature in FEATURE_NAMES}
            row.update({
                "label": label,
                "session_id": session_id,
                "host_id": host_id,
                "timestamp": snapshot.get("timestamp"),
                "notes": notes,
                "monitor_backend": file_events.get("backend", "disabled"),
            })
            rows.append(row)
            print(
                f"  [{step+1:03d}/{total_steps}] "
                f"CPU:{row['cpu_percent']:5.1f}% "
                f"Files:{int(row['file_modified_count']):3d} "
                f"Conns:{int(row['active_connections']):3d}"
            )
            time.sleep(interval)
    finally:
        monitor.stop()

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows -> {output_path}")
    return output_path


def main():
    args = parse_args()
    capture_session(
        label=args.label,
        duration=args.duration,
        interval=args.interval,
        session_id=args.session_id,
        host_id=args.host_id,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
