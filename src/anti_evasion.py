"""
anti_evasion.py — FIXED
Evidence accumulator now resets when confidence is consistently low.
AE detection requires multiple signals, not just accumulated evidence.
"""

import os, math, time, collections
import numpy as np
import datetime
import warnings; warnings.filterwarnings("ignore")

os.makedirs("logs", exist_ok=True)


class SlidingWindowAnalyzer:
    def __init__(self, window_seconds=60, check_interval=6):
        self.window_size = window_seconds // check_interval
        self.history     = collections.deque(maxlen=self.window_size)
        self.window_secs = window_seconds

    def update(self, snapshot):
        self.history.append({
            "cpu":        snapshot.get("cpu_percent", 0),
            "memory":     snapshot.get("memory_percent", 0),
            "file_mods":  snapshot.get("file_modified_count", 0),
            "file_dels":  snapshot.get("file_deleted_count", 0),
            "disk_write": snapshot.get("disk_write_rate", 0),
            "new_procs":  snapshot.get("new_process_count", 0),
            "net_sent":   snapshot.get("bytes_sent_rate", 0),
            "conns":      snapshot.get("active_connections", 0),
        })

        if len(self.history) < 5:
            return {"evasion_detected":False,"score":0.0,"reason":"Insufficient data",
                    "window_size":len(self.history),"total_file_mods":0,"avg_cpu":0}

        h = list(self.history)
        scores  = []
        reasons = []

        # Check 1: cumulative file modifications (must be high)
        total_mods = sum(x["file_mods"] for x in h)
        if total_mods > 100:
            scores.append(min(total_mods/300, 1.0) * 0.35)
            reasons.append(f"Cumulative file mods: {total_mods}")

        # Check 2: sustained very high CPU (>70% sustained)
        high_cpu = [x["cpu"] for x in h if x["cpu"] > 70]
        if len(high_cpu) >= len(h) * 0.6:
            scores.append(min(np.mean(high_cpu)/100, 1.0) * 0.25)
            reasons.append(f"Sustained high CPU: {np.mean(high_cpu):.1f}%")

        # Check 3: disk write increasing rapidly
        disk = [x["disk_write"] for x in h]
        if len(disk) >= 6:
            first = np.mean(disk[:3])
            last  = np.mean(disk[-3:])
            if last > first * 2.0 and last > 5e6:
                scores.append(min((last-first)/20e6, 1.0) * 0.20)
                reasons.append(f"Disk write spike: {last/1e6:.1f} MB/s")

        # Check 4: cumulative file deletions
        total_dels = sum(x["file_dels"] for x in h)
        if total_dels > 50:
            scores.append(min(total_dels/150, 1.0) * 0.20)
            reasons.append(f"Cumulative deletions: {total_dels}")

        total = sum(scores)
        return {
            "evasion_detected": total >= 0.50,
            "score":            round(total, 4),
            "reasons":          reasons,
            "window_size":      len(self.history),
            "window_seconds":   self.window_secs,
            "total_file_mods":  total_mods,
            "avg_cpu":          round(np.mean([x["cpu"] for x in h]), 1),
        }


class BehavioralDriftDetector:
    def __init__(self, baseline_samples=20):
        self.baseline_samples = baseline_samples
        self.baseline_data    = collections.deque(maxlen=baseline_samples)
        self.baseline_stats   = None
        self.is_calibrated    = False

    def update_baseline(self, snapshot):
        self.baseline_data.append({
            "cpu":    snapshot.get("cpu_percent", 0),
            "memory": snapshot.get("memory_percent", 0),
            "disk":   snapshot.get("disk_write_rate", 0),
            "net":    snapshot.get("bytes_sent_rate", 0),
            "conns":  snapshot.get("active_connections", 0),
        })
        if len(self.baseline_data) >= self.baseline_samples // 2:
            self._compute()
            self.is_calibrated = True

    def _compute(self):
        data = list(self.baseline_data)
        self.baseline_stats = {}
        for k in ["cpu","memory","disk","net","conns"]:
            vals = [d[k] for d in data]
            self.baseline_stats[k] = {"mean":np.mean(vals),"std":max(np.std(vals),0.01)}

    def detect_drift(self, snapshot):
        if not self.is_calibrated:
            return {"drift_detected":False,"score":0.0,
                    "reason":f"Calibrating ({len(self.baseline_data)}/{self.baseline_samples})",
                    "is_calibrated":False,"baseline_size":len(self.baseline_data)}
        cur = {"cpu":snapshot.get("cpu_percent",0),"memory":snapshot.get("memory_percent",0),
               "disk":snapshot.get("disk_write_rate",0),"net":snapshot.get("bytes_sent_rate",0),
               "conns":snapshot.get("active_connections",0)}
        zscores = {}
        reasons = []
        for k,v in cur.items():
            s = self.baseline_stats[k]
            z = (v - s["mean"]) / s["std"]
            zscores[k] = max(0, z)
            if z > 4.0:  # raised from 3.0 to 4.0 — only flag extreme deviations
                reasons.append(f"{k} drifted {z:.1f}σ (baseline:{s['mean']:.1f}, now:{v:.1f})")
        score = min(np.mean(list(zscores.values())) / 6.0, 1.0)  # raised denominator
        return {"drift_detected": score >= 0.5,"score":round(score,4),
                "reasons":reasons,"is_calibrated":True,"baseline_size":len(self.baseline_data)}


class EntropySpikeDetector:
    def __init__(self):
        self.history = collections.deque(maxlen=10)

    def update(self, snapshot):
        disk  = snapshot.get("disk_write_rate", 0)
        mods  = snapshot.get("file_modified_count", 0)
        dels  = snapshot.get("file_deleted_count", 0)
        cpu   = snapshot.get("cpu_percent", 0)
        procs = snapshot.get("new_process_count", 0)

        score = 0.0
        if disk > 20e6:   score += 0.30  # very high disk write
        if mods > 50:     score += 0.25  # many file modifications
        if dels > 20:     score += 0.20  # deleting originals
        if cpu > 80:      score += 0.15  # high CPU
        if procs > 10:    score += 0.10  # spawning many processes
        score = min(score, 1.0)

        self.history.append(score)
        avg  = np.mean(list(self.history))
        spike     = score > 0.60          # raised from 0.5
        sustained = avg > 0.50 and len(self.history) >= 5  # raised from 0.35

        reasons = []
        if spike:     reasons.append(f"Encryption spike: {score*100:.0f}%")
        if sustained: reasons.append(f"Sustained encryption: {avg*100:.0f}%")

        return {"encryption_detected":spike or sustained,"score":round(score,4),
                "avg_score":round(avg,4),"spike_detected":spike,
                "sustained_activity":sustained,"reasons":reasons}


class EvidenceAccumulator:
    """
    FIXED: Evidence now decays quickly when confidence is low.
    Resets when model confidence stays below 30% for 3+ checks.
    """
    def __init__(self, decay_rate=0.70, threshold=0.75):
        # decay_rate lowered (faster decay), threshold raised
        self.evidence  = 0.0
        self.decay     = decay_rate
        self.threshold = threshold
        self.low_conf_count = 0
        self.history   = collections.deque(maxlen=30)

    def update(self, confidence, sliding_score, drift_score, entropy_score):
        # If model confidence is low, decay evidence faster
        if confidence < 0.30:
            self.low_conf_count += 1
            if self.low_conf_count >= 3:
                self.evidence *= 0.5  # rapid reset when consistently normal
        else:
            self.low_conf_count = 0

        self.evidence *= self.decay

        # Only accumulate when multiple signals agree
        if sliding_score > 0.2 and entropy_score > 0.2:
            new_ev = (confidence*0.40 + sliding_score*0.25 +
                      drift_score*0.20 + entropy_score*0.15)
            self.evidence = min(self.evidence + new_ev, 1.0)

        self.history.append(self.evidence)
        return {"accumulated_evidence":round(self.evidence,4),"threshold":self.threshold,
                "alert":self.evidence >= self.threshold,
                "trend":"increasing" if len(self.history)>=3 and
                        self.history[-1]>self.history[-3] else "stable"}

    def reset(self):
        self.evidence = 0.0
        self.low_conf_count = 0


class AntiEvasionEngine:
    def __init__(self):
        self.sliding_window   = SlidingWindowAnalyzer(window_seconds=60, check_interval=6)
        self.drift_detector   = BehavioralDriftDetector(baseline_samples=20)
        self.entropy_detector = EntropySpikeDetector()
        self.accumulator      = EvidenceAccumulator(decay_rate=0.70, threshold=0.75)
        self.check_count      = 0

    def analyze(self, snapshot, model_confidence):
        self.check_count += 1

        if self.check_count <= 20:
            self.drift_detector.update_baseline(snapshot)

        sliding  = self.sliding_window.update(snapshot)
        drift    = self.drift_detector.detect_drift(snapshot)
        entropy  = self.entropy_detector.update(snapshot)
        evidence = self.accumulator.update(
            model_confidence,
            sliding["score"],
            drift["score"],
            entropy["score"]
        )

        evasion_score = max(
            sliding["score"],
            drift["score"],
            entropy["score"],
            evidence["accumulated_evidence"]
        )

        # Evasion detected ONLY when multiple signals agree
        evasion_detected = (
            (sliding["evasion_detected"] and entropy["encryption_detected"]) or
            (drift["drift_detected"] and sliding["evasion_detected"]) or
            evidence["alert"]
        )

        all_reasons = []
        all_reasons.extend(sliding.get("reasons", []))
        all_reasons.extend(drift.get("reasons", []))
        all_reasons.extend(entropy.get("reasons", []))

        enhanced_confidence = max(model_confidence, evasion_score * 0.7)

        return {
            "evasion_detected":    evasion_detected,
            "evasion_score":       round(evasion_score, 4),
            "enhanced_confidence": round(enhanced_confidence, 4),
            "model_confidence":    model_confidence,
            "sliding_window":      sliding,
            "behavioral_drift":    drift,
            "entropy_analysis":    entropy,
            "evidence":            evidence,
            "all_reasons":         all_reasons,
            "check_count":         self.check_count,
            "timestamp":           datetime.datetime.now().isoformat(),
        }

    def get_summary(self):
        return {"checks":self.check_count,
                "baseline_calibrated":self.drift_detector.is_calibrated,
                "accumulated_evidence":round(self.accumulator.evidence,4)}