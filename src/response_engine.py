"""
response_engine.py
When threat detected:
1. Find and terminate the most suspicious process
2. Send desktop notification
3. Log the incident
4. Return prevention steps
"""

import os
import psutil
import datetime
import platform
import subprocess

LOG_PATH = "logs/incidents.log"
os.makedirs("logs", exist_ok=True)

SAFE_PROCESSES = {
    "system", "svchost.exe", "explorer.exe", "winlogon.exe",
    "csrss.exe", "smss.exe", "lsass.exe", "services.exe",
    "wininit.exe", "dwm.exe", "registry", "conhost.exe",
    "fontdrvhost.exe", "spoolsv.exe", "audiodg.exe",
    "sihost.exe", "runtimebroker.exe", "taskhostw.exe",
    "searchhost.exe", "securityhealthservice.exe",
    "python.exe", "python3.exe", "code.exe", "cmd.exe",
    "powershell.exe", "windowsterminal.exe",
    "chrome.exe", "firefox.exe", "msedge.exe",
    "taskmgr.exe", "procexp.exe", "streamlit.exe",
}

PREVENTION_STEPS = [
    "Immediately disconnect from the internet to stop ransomware C&C communication.",
    "Do not pay any ransom — it does not guarantee file recovery.",
    "Run a full antivirus scan using Windows Defender or Malwarebytes.",
    "Check and restore files from your most recent backup.",
    "Change all passwords from a different, clean device.",
    "Update your operating system and all installed software immediately.",
    "Enable Windows Defender real-time protection if not already active.",
    "Avoid opening email attachments or links from unknown senders.",
    "Keep regular backups on an external drive disconnected when not in use.",
    "Enable two-factor authentication on all important accounts.",
]


def find_suspicious_process():
    candidates = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent",
                                      "memory_percent", "status"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name in SAFE_PROCESSES:
                continue
            cpu   = proc.info.get("cpu_percent") or 0
            mem   = proc.info.get("memory_percent") or 0
            score = cpu * 0.6 + mem * 0.4
            if score > 1:
                candidates.append({
                    "pid":    proc.info["pid"],
                    "name":   proc.info["name"],
                    "cpu":    cpu,
                    "memory": mem,
                    "score":  score,
                })
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]


def terminate_process(proc_info: dict) -> dict:
    try:
        proc = psutil.Process(proc_info["pid"])
        proc.kill()
        return {
            "success": True,
            "pid":     proc_info["pid"],
            "name":    proc_info["name"],
            "message": f"Process '{proc_info['name']}' (PID {proc_info['pid']}) terminated."
        }
    except psutil.NoSuchProcess:
        return {"success": False, "message": "Process already ended."}
    except psutil.AccessDenied:
        return {"success": False,
                "message": f"Access denied — run as Administrator to terminate '{proc_info['name']}'."}
    except Exception as e:
        return {"success": False, "message": str(e)}


def send_notification(title: str, message: str):
    try:
        if platform.system() == "Windows":
            try:
                from plyer import notification
                notification.notify(title=title, message=message,
                                    app_name="Ransomware Detection", timeout=10)
                return True
            except Exception:
                pass
            ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$notify = New-Object System.Windows.Forms.NotifyIcon
$notify.Icon = [System.Drawing.SystemIcons]::Warning
$notify.Visible = $true
$notify.ShowBalloonTip(10000, '{title}', '{message}', [System.Windows.Forms.ToolTipIcon]::Warning)
Start-Sleep -Seconds 3
$notify.Dispose()
"""
            subprocess.Popen(
                ["powershell", "-WindowStyle", "Hidden", "-Command", ps_script],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True
    except Exception:
        return False


def log_incident(confidence: float, votes: int, terminated: dict = None):
    timestamp = datetime.datetime.now().isoformat()
    line = (f"\n{'='*60}\n"
            f"INCIDENT DETECTED\n"
            f"Time       : {timestamp}\n"
            f"Confidence : {confidence*100:.1f}%\n"
            f"Model votes: {votes}/5\n")
    if terminated:
        line += f"Action     : {terminated.get('message','N/A')}\n"
    line += f"{'='*60}\n"
    with open(LOG_PATH, "a") as f:
        f.write(line)
    print(f"  Incident logged -> {LOG_PATH}")


def respond_to_threat(result: dict, auto_terminate: bool = True) -> dict:
    conf  = result["confidence"]
    votes = result["vote_count"]

    response = {
        "timestamp":         datetime.datetime.now().isoformat(),
        "confidence":        conf,
        "votes":             votes,
        "process_found":     None,
        "termination":       None,
        "notification_sent": False,
        "prevention_steps":  PREVENTION_STEPS,
    }

    print("\n" + "!"*55)
    print("  !! RANSOMWARE THREAT DETECTED !!")
    print("!"*55)
    print(f"  Confidence : {conf*100:.1f}%")
    print(f"  Votes      : {votes}/5 models flagged")

    if auto_terminate:
        proc = find_suspicious_process()
        if proc:
            response["process_found"] = proc
            print(f"\n  Suspicious process: {proc['name']} (PID {proc['pid']}) CPU:{proc['cpu']:.1f}%")
            term_result = terminate_process(proc)
            response["termination"] = term_result
            print(f"  Action: {term_result['message']}")
        else:
            print("  No suspicious process found.")

    notif_msg = (f"Confidence: {conf*100:.1f}% | {votes}/5 models flagged. "
                 f"{'Process terminated. ' if response.get('termination',{}).get('success') else ''}"
                 "Check your system immediately!")
    sent = send_notification("⚠️ RANSOMWARE THREAT DETECTED", notif_msg)
    response["notification_sent"] = sent
    print(f"  Notification: {'Sent' if sent else 'Failed'}")

    log_incident(conf, votes, response.get("termination"))

    print("!"*55 + "\n")
    return response