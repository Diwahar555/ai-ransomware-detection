"""
Detection History Tab — Streamlit Dashboard Integration
────────────────────────────────────────────────────────
File: src/detection_history.py

Stores all past threat detections and provides
a full history table, filters, stats, and CSV export.
"""

import json
import os
import threading
from datetime import datetime

HISTORY_FILE = r"V:\New folder (3)\Ransomeware\logs\detection_history.json"

class DetectionHistory:
    """
    Persistent detection history — survives dashboard restarts.
    Thread-safe. Saves to JSON log file automatically.
    """

    def __init__(self, history_file=HISTORY_FILE):
        self.history_file = history_file
        self._lock        = threading.Lock()
        self.records      = self._load()

    def _load(self):
        """Load existing history from JSON file."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[HISTORY] Could not load: {e}")
        return []

    def _save(self):
        """Save current history to JSON file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.records, f, indent=2)
        except Exception as e:
            print(f"[HISTORY] Could not save: {e}")

    def add(self, process_name, pid, confidence, threat_level,
            action_taken, model_scores=None):
        """
        Add a new detection record.

        Args:
            process_name : Name of suspicious process
            pid          : Process ID
            confidence   : Final ensemble confidence (0.0 - 1.0)
            threat_level : 'LOW' / 'MEDIUM' / 'HIGH' / 'CRITICAL'
            action_taken : e.g. 'Quarantine + Network Block'
            model_scores : dict with individual model scores (optional)
        """
        record = {
            "id":           len(self.records) + 1,
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "process_name": process_name,
            "pid":          pid,
            "confidence":   round(confidence * 100, 1),
            "threat_level": threat_level,
            "action_taken": action_taken,
            "model_scores": model_scores or {},
            "resolved":     False,
        }
        with self._lock:
            self.records.append(record)
            self._save()
        return record

    def mark_resolved(self, record_id):
        """Mark a detection as resolved."""
        with self._lock:
            for r in self.records:
                if r["id"] == record_id:
                    r["resolved"] = True
                    break
            self._save()

    def clear_all(self):
        """Clear all history."""
        with self._lock:
            self.records = []
            self._save()

    def get_all(self):
        """Return all records (newest first)."""
        with self._lock:
            return list(reversed(self.records))

    def get_stats(self):
        """Return summary statistics."""
        with self._lock:
            total    = len(self.records)
            critical = sum(1 for r in self.records if r["threat_level"] == "CRITICAL")
            high     = sum(1 for r in self.records if r["threat_level"] == "HIGH")
            medium   = sum(1 for r in self.records if r["threat_level"] == "MEDIUM")
            low      = sum(1 for r in self.records if r["threat_level"] == "LOW")
            resolved = sum(1 for r in self.records if r.get("resolved"))
            avg_conf = (
                round(sum(r["confidence"] for r in self.records) / total, 1)
                if total else 0.0
            )
            unique_processes = len(set(r["process_name"] for r in self.records))
            return {
                "total": total, "critical": critical, "high": high,
                "medium": medium, "low": low, "resolved": resolved,
                "avg_confidence": avg_conf,
                "unique_processes": unique_processes,
            }


# Singleton instance
history = DetectionHistory()