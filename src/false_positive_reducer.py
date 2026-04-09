"""
false_positive_reducer.py
Improvement 2 — False Positive Reduction
- Multi-stage confirmation (3 consecutive detections before alert)
- Trusted process whitelist
- Time-based analysis (2AM spike = suspicious, 2PM spike = normal)
- Confidence smoothing using rolling average
"""

import datetime
import collections
import psutil

# ── Trusted processes — never flagged ────────────────────────────────────────
TRUSTED_PROCESSES = {
    # System
    "system", "svchost.exe", "explorer.exe", "winlogon.exe",
    "csrss.exe", "smss.exe", "lsass.exe", "services.exe",
    "wininit.exe", "dwm.exe", "registry", "conhost.exe",
    "fontdrvhost.exe", "spoolsv.exe", "audiodg.exe",
    "sihost.exe", "runtimebroker.exe", "taskhostw.exe",
    "searchhost.exe", "securityhealthservice.exe",
    # Development tools
    "code.exe", "cmd.exe",
    "powershell.exe", "windowsterminal.exe", "git.exe",
    "node.exe", "npm.cmd", "pip.exe",
    # Browsers
    "chrome.exe", "firefox.exe", "msedge.exe", "opera.exe",
    "brave.exe", "iexplore.exe",
    # Office
    "winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe",
    "onenote.exe", "teams.exe", "onedrive.exe",
    # Media
    "vlc.exe", "wmplayer.exe", "spotify.exe",
    # Security
    "msmpeng.exe", "antimalware.exe", "defender.exe",
    "taskmgr.exe", "procexp.exe",
    # Streaming
    "streamlit.exe",
}

# ── Time-based thresholds ─────────────────────────────────────────────────────
# During business hours (9AM-6PM) CPU spikes are more normal
# Late night (11PM-6AM) any spike is more suspicious
def get_time_adjusted_threshold(base_threshold: float) -> float:
    hour = datetime.datetime.now().hour

    if 0 <= hour < 6:
        # Late night -- slightly more sensitive (reduced from -0.15)
        return max(base_threshold - 0.05, 0.45)
    elif 6 <= hour < 9:
        # Early morning -- normal
        return base_threshold
    elif 9 <= hour < 18:
        # Business hours -- normal threshold
        return base_threshold
    elif 18 <= hour < 22:
        # Evening -- normal threshold
        return base_threshold
    else:
        # Night -- slightly more sensitive
        return max(base_threshold - 0.05, 0.45)


# ── Multi-stage confirmation ──────────────────────────────────────────────────
class MultiStageConfirmation:
    """
    Requires N consecutive threat detections before triggering response.
    Prevents single-spike false positives.
    """
    def __init__(self, required_consecutive: int = 3, window_size: int = 10):
        self.required   = required_consecutive
        self.history    = collections.deque(maxlen=window_size)
        self.consecutive= 0
        self.confirmed  = False

    def update(self, is_threat: bool, confidence: float) -> dict:
        self.history.append({"threat": is_threat, "confidence": confidence,
                              "time": datetime.datetime.now().isoformat()})

        if is_threat:
            self.consecutive += 1
        else:
            self.consecutive = 0
            self.confirmed   = False

        # Confirm only after N consecutive detections
        if self.consecutive >= self.required:
            self.confirmed = True

        # Rolling average confidence
        recent = list(self.history)[-self.required:]
        avg_conf = sum(r["confidence"] for r in recent) / max(len(recent), 1)

        return {
            "confirmed":          self.confirmed,
            "consecutive_count":  self.consecutive,
            "required":           self.required,
            "rolling_confidence": round(avg_conf, 4),
            "history_size":       len(self.history),
            "stage":              f"{min(self.consecutive, self.required)}/{self.required}",
        }

    def reset(self):
        self.consecutive = 0
        self.confirmed   = False


# ── Confidence smoother ───────────────────────────────────────────────────────
class ConfidenceSmoother:
    """
    Smooths confidence scores using exponential moving average.
    Prevents single-spike alerts from noise.
    """
    def __init__(self, alpha: float = 0.4):
        self.alpha   = alpha   # smoothing factor (0=very smooth, 1=no smoothing)
        self.smoothed = None

    def update(self, raw_confidence: float) -> float:
        if self.smoothed is None:
            self.smoothed = raw_confidence
        else:
            self.smoothed = self.alpha * raw_confidence + (1 - self.alpha) * self.smoothed
        return round(self.smoothed, 4)


# ── Trusted process checker ───────────────────────────────────────────────────
def is_trusted_process(process_name: str) -> bool:
    return process_name.lower() in TRUSTED_PROCESSES


def get_untrusted_high_cpu_processes(cpu_threshold: float = 10.0) -> list:
    """Returns list of high-CPU processes that are NOT trusted."""
    untrusted = []
    try:
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cpu  = proc.info.get("cpu_percent") or 0
                if cpu >= cpu_threshold and name not in TRUSTED_PROCESSES:
                    untrusted.append({
                        "pid":    proc.info["pid"],
                        "name":   proc.info["name"],
                        "cpu":    cpu,
                        "memory": proc.info.get("memory_percent") or 0,
                    })
            except Exception:
                pass
    except Exception:
        pass
    untrusted.sort(key=lambda x: x["cpu"], reverse=True)
    return untrusted


# ── Context analyzer ─────────────────────────────────────────────────────────
def analyze_context(snapshot: dict, confidence: float, raw_confidence: float = 0.0) -> dict:
    """
    Analyze context to reduce false positives:
    - Is it a known high-CPU time? (backup, antivirus scan)
    - Is the high CPU from a trusted process?
    - Is the network spike from a trusted app?
    raw_confidence: the ensemble's raw output BEFORE smoothing.
    When the raw model is already highly confident (>=65%), we
    significantly dampen the file-churn penalty — fileless and
    process-injection attacks produce no file events but the
    ensemble still detects CPU/memory/process anomalies correctly.
    """
    hour       = datetime.datetime.now().hour
    cpu        = snapshot.get("cpu_percent", 0)
    disk_write = snapshot.get("disk_write_rate", 0)
    new_procs  = snapshot.get("new_process_count", 0)
    file_mods  = snapshot.get("file_modified_count", 0)
    file_creates = snapshot.get("file_created_count", 0)
    file_deletes = snapshot.get("file_deleted_count", 0)
    active_conns = snapshot.get("active_connections", 0)
    bytes_sent = snapshot.get("bytes_sent_rate", 0)
    bytes_recv = snapshot.get("bytes_recv_rate", 0)
    file_activity = file_mods + file_creates + file_deletes

    reasons = []
    penalty = 0.0   # reduce confidence by this amount
    suspicion_score = 0.0

    # Check if high CPU is from trusted processes (raised threshold: 25% not 20%)
    untrusted_procs = get_untrusted_high_cpu_processes(cpu_threshold=25)
    if untrusted_procs:
        suspicion_score += min(0.25, 0.08 * len(untrusted_procs))
    if file_activity >= 20:
        suspicion_score += 0.45
    elif file_activity >= 5:
        suspicion_score += 0.20
    if disk_write >= 10e6:
        suspicion_score += 0.30
    elif disk_write >= 3e6:
        suspicion_score += 0.12
    if new_procs >= 10:
        suspicion_score += 0.15
    elif new_procs >= 4:
        suspicion_score += 0.07
    if bytes_sent >= 2e5 or bytes_recv >= 5e5 or active_conns >= 140:
        suspicion_score += 0.10

    # Very high CPU with no actually suspicious processes → likely antivirus/compiler/update
    if cpu > 50 and len(untrusted_procs) == 0:
        reasons.append("High CPU but no untrusted processes found")
        penalty += 0.20

    # Scheduled tasks run at specific times
    if hour in [0, 2, 3] and disk_write > 5e6:
        reasons.append("High disk write at scheduled backup hour")
        penalty += 0.10

    # Windows Update typically runs at night
    if hour in [2, 3, 4] and cpu > 50:
        reasons.append("High CPU at Windows Update hour")
        penalty += 0.10

    # Very few new processes is less suspicious
    if new_procs <= 3 and cpu < 80:
        reasons.append("Low new process count")
        penalty += 0.08

    # Quiet file-system is the strongest single signal against ransomware —
    # BUT only when the model ensemble itself is not already confident.
    # When raw_confidence >= 0.65 the models are detecting CPU/process/network
    # anomalies (fileless attack patterns); don't suppress that with file-churn rules.
    file_churn_scale = max(0.0, 1.0 - (raw_confidence - 0.50) / 0.25) if raw_confidence >= 0.50 else 1.0
    file_churn_scale = min(file_churn_scale, 1.0)

    if file_activity == 0 and disk_write < 1e6:
        reasons.append("No meaningful file churn detected")
        penalty += 0.60 * file_churn_scale
    elif file_activity == 0 and disk_write < 5e6:
        reasons.append("No file changes detected")
        penalty += 0.45 * file_churn_scale
    elif file_activity <= 2 and disk_write < 3e6:
        reasons.append("Very low file activity")
        penalty += 0.40 * file_churn_scale
    elif file_activity <= 5 and disk_write < 8e6:
        reasons.append("Low file activity")
        penalty += 0.20 * file_churn_scale

    if len(untrusted_procs) == 0 and new_procs <= 5:
        reasons.append("No suspicious process activity")
        penalty += 0.12

    # Normal-range network: no big burst
    if active_conns < 150 and bytes_sent < 1.5e5 and bytes_recv < 3e5:
        reasons.append("No unusual network burst")
        penalty += 0.08

    adjusted_confidence = max(0.0, confidence - penalty)
    suspicion_score = min(round(suspicion_score, 4), 1.0)

    return {
        "original_confidence": confidence,
        "adjusted_confidence": round(adjusted_confidence, 4),
        "penalty_applied":     round(penalty, 4),
        "suspicion_score":     suspicion_score,
        "reasons":             reasons,
        "untrusted_processes": untrusted_procs[:5],
        "time_hour":           hour,
    }


# ── Main FP reducer ───────────────────────────────────────────────────────────
class FalsePositiveReducer:
    """
    Complete false positive reduction pipeline.
    Combines all techniques into one easy interface.
    """
    def __init__(self, required_consecutive: int = 3,
                 smoothing_alpha: float = 0.4,
                 base_threshold: float = 0.5):
        self.confirmer  = MultiStageConfirmation(required_consecutive)
        self.smoother   = ConfidenceSmoother(smoothing_alpha)
        self.base_threshold = base_threshold

    def process(self, raw_result: dict, snapshot: dict) -> dict:
        """
        Process a raw prediction result through all FP reduction steps.
        Returns enhanced result with FP reduction applied.
        """
        raw_confidence = raw_result["confidence"]

        # Step 1: Smooth confidence
        smoothed_conf = self.smoother.update(raw_confidence)

        # Step 2: Context analysis — pass raw confidence so file-churn
        # penalty is scaled down when the ensemble is already alarmed.
        context = analyze_context(snapshot, smoothed_conf, raw_confidence)
        adj_conf = context["adjusted_confidence"]
        suspicion_score = context["suspicion_score"]

        # Step 3: Time-adjusted threshold
        threshold = get_time_adjusted_threshold(self.base_threshold)

        # Step 4: Only advance confirmation when there is STRONG supporting behavioral evidence.
        evidence_gate = suspicion_score >= 0.35
        is_threat_adj = adj_conf >= threshold and evidence_gate

        # Step 5: Multi-stage confirmation
        confirmation = self.confirmer.update(is_threat_adj, adj_conf)

        # Final decision — only confirmed threats trigger response
        final_threat = confirmation["confirmed"]

        return {
            "is_threat":            final_threat,
            "raw_confidence":       raw_confidence,
            "smoothed_confidence":  smoothed_conf,
            "adjusted_confidence":  adj_conf,
            "threshold_used":       threshold,
            "stage":                confirmation["stage"],
            "consecutive_count":    confirmation["consecutive_count"],
            "confirmed":            final_threat,
            "context_reasons":      context["reasons"],
            "penalty_applied":      context["penalty_applied"],
            "suspicion_score":      suspicion_score,
            "evidence_gate":        evidence_gate,
            "untrusted_processes":  context["untrusted_processes"],
            "votes":                raw_result["votes"],
            "probabilities":        raw_result["probabilities"],
            "vote_count":           raw_result["vote_count"],
            "timestamp":            raw_result["timestamp"],
            "snapshot":             snapshot,
        }

    def reset(self):
        self.confirmer.reset()
