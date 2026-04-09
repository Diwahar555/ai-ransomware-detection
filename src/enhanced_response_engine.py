"""
enhanced_response_engine.py
Runtime response helpers for the ransomware dashboard.
"""

import datetime
import json
import os
import platform
import subprocess
import warnings

import psutil

warnings.filterwarnings("ignore")

LOG_PATH = "logs/incidents.log"
QUARANTINE_DB = "logs/quarantine.json"
os.makedirs("logs", exist_ok=True)

SAFE_PROCESSES = {
    "system", "system idle process", "idle", "registry",
    "svchost.exe", "explorer.exe", "winlogon.exe", "csrss.exe",
    "smss.exe", "lsass.exe", "services.exe", "wininit.exe",
    "dwm.exe", "conhost.exe", "fontdrvhost.exe", "spoolsv.exe",
    "audiodg.exe", "sihost.exe", "runtimebroker.exe", "taskhostw.exe",
    "searchhost.exe", "securityhealthservice.exe", "ntoskrnl.exe",
    "memory compression", "memcompression",
    "phoneexperiencehost.exe", "startmenuexperiencehost.exe",
    "shellexperiencehost.exe", "searchapp.exe", "searchindexer.exe",
    "ctfmon.exe", "dllhost.exe", "backgroundtaskhost.exe",
    "applicationframehost.exe", "systemsettings.exe",
    "textinputhost.exe", "lockapp.exe", "logonui.exe",
    "userinit.exe", "taskmgr.exe", "msiexec.exe",
    "wuauclt.exe", "tiworker.exe", "trustedinstaller.exe",
    "compattelrunner.exe", "wsappx.exe", "wmpnetwk.exe",
    "lsaiso.exe", "windefend.exe",
    "chrome.exe", "firefox.exe", "msedge.exe", "brave.exe",
    "opera.exe", "iexplore.exe", "chromium.exe",
    "msedgewebview.exe", "msedgewebview2.exe",
    "browserhost.exe", "browser_broker.exe",
    "chromedriver.exe", "geckodriver.exe",
    "msmpeng.exe", "antimalware service executable",
    "mssense.exe", "nissrv.exe", "securityhealthsystray.exe",
    "mbam.exe", "mbamtray.exe",
    "python.exe", "python3.exe", "pythonw.exe",
    "code.exe", "code - insiders.exe",
    "cmd.exe", "powershell.exe", "pwsh.exe",
    "windowsterminal.exe", "wt.exe",
    "git.exe", "git-remote-https.exe",
    "node.exe", "npm.cmd", "streamlit.exe",
    "winword.exe", "excel.exe", "powerpnt.exe",
    "outlook.exe", "onenote.exe", "teams.exe",
    "onedrive.exe", "msteams.exe", "update.exe",
    "officeclicktorun.exe", "appvshnotify.exe",
    "servicehost.exe", "mc-dad.exe", "mc-agent.exe",
    "igfxem.exe", "igfxtray.exe", "igfxhk.exe",
    "nvcontainer.exe", "nvtelemetry.exe", "nvidia.exe",
    "amdow.exe", "radeon.exe",
    "realtek.exe", "rtkauduservice64.exe",
    "wlanext.exe", "wlidsvc.exe",
    "vlc.exe", "wmplayer.exe", "spotify.exe",
    "groove.exe", "movies.exe",
    "procexp.exe", "procexp64.exe", "autoruns.exe",
    "perfmon.exe", "resmon.exe",
}

SUSPICIOUS_NAME_HINTS = (
    "encrypt", "locker", "ransom", "cipher", "decrypt",
    "payload", "dropper", "wiper", "scrambler",
)
SUSPICIOUS_PARENT_HINTS = {
    "powershell.exe", "pwsh.exe", "cmd.exe",
    "wscript.exe", "cscript.exe", "mshta.exe",
    "python.exe", "pythonw.exe",
}
PROCESS_REVIEW_SCORE = 6
AUTO_QUARANTINE_SCORE = 8

SENSITIVE_FOLDERS = [
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Pictures"),
    os.path.expanduser("~/Downloads"),
]

PREVENTION_STEPS = [
    "Immediately disconnect from the internet to stop ransomware C&C communication.",
    "Do not pay any ransom because it does not guarantee file recovery.",
    "Run a full antivirus scan using Windows Defender or Malwarebytes.",
    "Check and restore files from your most recent backup.",
    "Change all passwords from a different, clean device.",
    "Update your operating system and all installed software immediately.",
    "Enable Windows Defender real-time protection if not already active.",
    "Avoid opening email attachments or links from unknown senders.",
    "Keep regular backups on an external drive disconnected when not in use.",
    "Enable two-factor authentication on all important accounts.",
]


def _normalize_path(path):
    if not path:
        return ""
    return os.path.normcase(os.path.normpath(path))


def _path_under(path, roots):
    norm_path = _normalize_path(path)
    if not norm_path:
        return False
    for root in roots:
        norm_root = _normalize_path(root)
        if norm_root and (norm_path == norm_root or norm_path.startswith(norm_root + os.sep)):
            return True
    return False


def trusted_install_roots():
    roots = [
        os.environ.get("SystemRoot", r"C:\Windows"),
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ]
    return [root for root in roots if root]


def user_writable_roots():
    home = os.path.expanduser("~")
    roots = [
        os.path.join(home, "AppData", "Local", "Temp"),
        os.path.join(home, "AppData", "Roaming"),
        os.path.join(home, "Downloads"),
        os.path.join(home, "Desktop"),
        os.path.join(home, "Documents"),
        os.path.join(home, "Pictures"),
    ]
    return [root for root in roots if root]


def is_user_writable_path(path):
    return _path_under(path, user_writable_roots())


def is_trusted_install_path(path):
    return _path_under(path, trusted_install_roots())


def get_process_details(proc):
    details = {
        "pid": proc.info.get("pid", 0),
        "name": proc.info.get("name") or "unknown",
        "cpu": float(proc.info.get("cpu_percent") or 0),
        "memory": float(proc.info.get("memory_percent") or 0),
        "exe": "",
        "cmdline": "",
        "username": "",
        "parent_name": "",
    }

    try:
        details["exe"] = proc.exe() or ""
    except Exception:
        details["exe"] = ""

    try:
        cmdline = proc.cmdline() or []
        details["cmdline"] = " ".join(cmdline[:8])
    except Exception:
        details["cmdline"] = ""

    try:
        details["username"] = proc.username() or ""
    except Exception:
        details["username"] = ""

    try:
        parent = proc.parent()
        details["parent_name"] = parent.name().lower() if parent else ""
    except Exception:
        details["parent_name"] = ""

    return details


def assess_process_risk(proc):
    details = get_process_details(proc)
    name = details["name"].lower().strip()
    exe = details["exe"]
    cmdline = details["cmdline"].lower()
    cpu = details["cpu"]
    memory = details["memory"]

    score = 0
    reasons = []

    if any(hint in name for hint in SUSPICIOUS_NAME_HINTS):
        score += 5
        reasons.append("Suspicious process name")

    if exe and is_user_writable_path(exe):
        score += 4
        reasons.append("Running from a user-writable path")
    elif exe and is_trusted_install_path(exe):
        score -= 3

    if details["parent_name"] in SUSPICIOUS_PARENT_HINTS:
        score += 2
        reasons.append(f"Spawned by {details['parent_name']}")

    if any(token in cmdline for token in ["encrypt", ".locked", ".encrypted", "/delete", "vssadmin"]):
        score += 3
        reasons.append("Suspicious command line activity")

    if cpu >= 70:
        score += 5
        reasons.append("Very high CPU")
    elif cpu >= 35:
        score += 3
        reasons.append("High CPU")
    elif cpu >= 15:
        score += 1
        reasons.append("Elevated CPU")

    if memory >= 10:
        score += 2
        reasons.append("High memory use")
    elif memory >= 5:
        score += 1
        reasons.append("Elevated memory use")

    username = details["username"].lower()
    if username and "system" not in username and "service" not in username:
        score += 1

    details["risk_score"] = max(score, 0)
    details["reasons"] = reasons
    details["path_risky"] = bool(exe and is_user_writable_path(exe))
    details["path_trusted"] = bool(exe and is_trusted_install_path(exe))
    return details


def isolate_network(block=True):
    rule_name = "RansomwareDetectionBlock"
    result = {"success": False, "action": "block" if block else "unblock", "message": ""}
    if platform.system() != "Windows":
        result["message"] = "Only supported on Windows"
        return result
    try:
        cmd = (
            f'netsh advfirewall firewall add rule name="{rule_name}" dir=out action=block protocol=any enable=yes'
            if block else
            f'netsh advfirewall firewall delete rule name="{rule_name}"'
        )
        ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        result["success"] = ret.returncode == 0
        result["message"] = (
            f"Network {'blocked' if block else 'unblocked'}"
            if result["success"] else (ret.stderr.strip() or ret.stdout.strip() or "Firewall command failed.")
        )
    except Exception as exc:
        result["message"] = str(exc)
    print(f"  Network: {result['message']}")
    return result


def protect_files(protect=True):
    result = {
        "success": True,
        "protected": [],
        "failed": [],
        "action": "protect" if protect else "restore",
        "message": "",
    }
    for folder in SENSITIVE_FOLDERS:
        if not os.path.exists(folder):
            continue
        try:
            cmd = (
                ["icacls", folder, "/deny", "Everyone:(W,D,DC)", "/T", "/Q"]
                if protect else
                ["icacls", folder, "/remove:d", "Everyone", "/T", "/Q"]
            )
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                result["protected"].append(folder)
            else:
                result["success"] = False
                error = proc.stderr.strip() or proc.stdout.strip() or f"icacls exited with {proc.returncode}"
                result["failed"].append({"folder": folder, "error": error})
        except Exception as exc:
            result["success"] = False
            result["failed"].append({"folder": folder, "error": str(exc)})

    if result["failed"]:
        result["message"] = f"{len(result['failed'])} folder operations failed."
    elif result["protected"]:
        verb = "Protected" if protect else "Restored"
        result["message"] = f"{verb} {len(result['protected'])} folders."
    else:
        result["message"] = "No sensitive folders found."
    return result


def find_suspicious_process(min_score=PROCESS_REVIEW_SCORE):
    candidates = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            name = (proc.info.get("name") or "").lower().strip()
            pid = proc.info.get("pid", 0)
            if pid <= 4:
                continue
            if name in SAFE_PROCESSES:
                continue
            cpu = proc.info.get("cpu_percent") or 0
            if cpu > 200:
                continue

            candidate = assess_process_risk(proc)
            if candidate["risk_score"] >= min_score:
                candidates.append(candidate)
        except Exception:
            pass
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (x["risk_score"], x["cpu"], x["memory"]), reverse=True)[0]


def quarantine_process(proc_info, kill=False):
    result = {
        "success": False,
        "action": "kill" if kill else "suspend",
        "pid": proc_info["pid"],
        "name": proc_info["name"],
        "message": "",
    }
    try:
        proc = psutil.Process(proc_info["pid"])
        proc.kill() if kill else proc.suspend()
        result["success"] = True
        result["message"] = (
            f"'{proc_info['name']}' (PID {proc_info['pid']}) "
            f"{'terminated' if kill else 'suspended'}."
        )
        if not kill:
            db = []
            if os.path.exists(QUARANTINE_DB):
                try:
                    with open(QUARANTINE_DB, "r") as handle:
                        db = json.load(handle)
                except Exception:
                    db = []
            db.append({
                "pid": proc_info["pid"],
                "name": proc_info["name"],
                "time": datetime.datetime.now().isoformat(),
                "action": "suspended",
            })
            with open(QUARANTINE_DB, "w") as handle:
                json.dump(db, handle, indent=2)
    except psutil.NoSuchProcess:
        result["success"] = True
        result["message"] = "Already ended."
    except psutil.AccessDenied:
        result["message"] = "Access denied. Run as administrator."
    except Exception as exc:
        result["message"] = str(exc)
    print(f"  Quarantine: {result['message']}")
    return result


def restore_quarantined_process(pid):
    result = {"success": False, "message": ""}
    try:
        psutil.Process(pid).resume()
        result["success"] = True
        result["message"] = f"PID {pid} resumed."
        if os.path.exists(QUARANTINE_DB):
            with open(QUARANTINE_DB, "r") as handle:
                db = json.load(handle)
            db = [entry for entry in db if entry["pid"] != pid]
            with open(QUARANTINE_DB, "w") as handle:
                json.dump(db, handle, indent=2)
    except Exception as exc:
        result["message"] = str(exc)
    return result


def get_quarantine_list():
    if not os.path.exists(QUARANTINE_DB):
        return []
    try:
        with open(QUARANTINE_DB, "r") as handle:
            return json.load(handle)
    except Exception:
        return []


def send_notification(title, message):
    try:
        if platform.system() == "Windows":
            try:
                from plyer import notification
                notification.notify(
                    title=title,
                    message=message,
                    app_name="Ransomware Detection",
                    timeout=10,
                )
                return True
            except Exception:
                pass
            ps_script = f"""Add-Type -AssemblyName System.Windows.Forms
$n=New-Object System.Windows.Forms.NotifyIcon
$n.Icon=[System.Drawing.SystemIcons]::Warning;$n.Visible=$true
$n.ShowBalloonTip(8000,'{title}','{message}',[System.Windows.Forms.ToolTipIcon]::Warning)
Start-Sleep -Seconds 3;$n.Dispose()"""
            subprocess.Popen(
                ["powershell", "-WindowStyle", "Hidden", "-Command", ps_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
    except Exception:
        return False
    return False


def log_incident(confidence, votes, actions):
    net_ok = isinstance(actions.get("network"), dict) and actions["network"].get("success", False)
    protected = (actions.get("file_protection") or {}).get("protected", [])
    qmsg = (actions.get("quarantine") or {}).get("message", "N/A")
    line = (
        f"\n{'=' * 60}\nINCIDENT DETECTED\n"
        f"Time: {datetime.datetime.now().isoformat()}\n"
        f"Confidence: {confidence * 100:.1f}%\nVotes: {votes}/5\n"
        f"Network: {net_ok}\nFiles: {len(protected)}\n"
        f"Process: {qmsg}\nNotification: {actions.get('notified', False)}\n{'=' * 60}\n"
    )
    with open(LOG_PATH, "a") as handle:
        handle.write(line)


def restore_all():
    print("\n-- Restoring all --")
    results = {
        "network": isolate_network(False),
        "files": protect_files(False),
        "processes": [],
    }
    for entry in get_quarantine_list():
        results["processes"].append(restore_quarantined_process(entry["pid"]))
    return results


def respond_to_threat(result, auto_terminate=True, isolate_net=False,
                      protect_files_flag=False, quarantine_mode=True):
    conf = result.get("confidence", 0) if isinstance(result, dict) else 0
    votes = result.get("vote_count", 0) if isinstance(result, dict) else 0
    response = {
        "timestamp": datetime.datetime.now().isoformat(),
        "confidence": conf,
        "votes": votes,
        "process_found": None,
        "process_review": None,
        "quarantine": None,
        "network": None,
        "file_protection": None,
        "notification_sent": False,
        "prevention_steps": PREVENTION_STEPS,
    }

    print(f"\n{'!' * 55}\n  THREAT DETECTED | Confidence:{conf * 100:.1f}% Votes:{votes}/5\n{'!' * 55}")

    if isolate_net:
        response["network"] = isolate_network(True)
    if protect_files_flag:
        response["file_protection"] = protect_files(True)
    if auto_terminate:
        proc = find_suspicious_process()
        if proc:
            response["process_found"] = proc
            response["process_review"] = {
                "score": proc["risk_score"],
                "reasons": proc.get("reasons", []),
                "message": f"Candidate {proc['name']} scored {proc['risk_score']} for containment review.",
            }
            print(
                f"  Found: {proc['name']} (PID {proc['pid']}) "
                f"CPU:{proc['cpu']:.1f}% Risk:{proc['risk_score']}"
            )
            if proc["risk_score"] >= AUTO_QUARANTINE_SCORE:
                response["quarantine"] = quarantine_process(proc, kill=not quarantine_mode)
            else:
                response["quarantine"] = {
                    "success": False,
                    "action": "skipped",
                    "pid": proc["pid"],
                    "name": proc["name"],
                    "message": (
                        f"Skipped automatic containment: risk score {proc['risk_score']} "
                        f"is below the auto-quarantine threshold of {AUTO_QUARANTINE_SCORE}."
                    ),
                }
                print("  Automatic containment skipped pending manual review.")
        else:
            print("  No suspicious process found.")

    actions = []
    if isinstance(response.get("network"), dict) and response["network"].get("success"):
        actions.append("Network blocked")
    if isinstance(response.get("quarantine"), dict) and response["quarantine"].get("success"):
        actions.append("Process quarantined")

    sent = send_notification(
        "RANSOMWARE THREAT",
        f"Confidence:{conf * 100:.1f}% | {votes}/5 models. {' | '.join(actions)}. Check dashboard!",
    )
    response["notification_sent"] = sent
    log_incident(
        conf,
        votes,
        {
            "network": response.get("network"),
            "file_protection": response.get("file_protection"),
            "quarantine": response.get("quarantine"),
            "notified": sent,
        },
    )
    print(f"{'!' * 55}\n")
    return response
