"""
file_activity_monitor.py
Lightweight file activity tracking for live detection and data capture.
Uses watchdog when available, with a polling fallback.
"""

import os
import time
import threading
from collections import defaultdict

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:  # pragma: no cover - optional dependency
    FileSystemEventHandler = None
    Observer = None


DEFAULT_WATCH_PATHS = [
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Pictures"),
    os.path.expanduser("~/Downloads"),
    os.path.expanduser("~/attack_test_folder"), # Direct home-folder fallback
    os.getcwd(), # The local project directory
]


def get_default_watch_paths():
    seen = set()
    paths = []
    for path in DEFAULT_WATCH_PATHS:
        norm = os.path.normpath(path)
        if os.path.exists(norm) and norm not in seen:
            seen.add(norm)
            paths.append(norm)
    return paths


class _WatchdogCounter(FileSystemEventHandler):
    def __init__(self, monitor):
        self.monitor = monitor

    def _handle(self, event_name, event):
        if getattr(event, "is_directory", False):
            return
        self.monitor._increment(event_name)

    def on_created(self, event):
        self._handle("created", event)

    def on_modified(self, event):
        self._handle("modified", event)

    def on_deleted(self, event):
        self._handle("deleted", event)

    def on_moved(self, event):
        self._handle("deleted", event)
        self._handle("created", event)


class FileActivityMonitor:
    def __init__(self, watch_paths=None, recursive=True, force_polling=False):
        self.watch_paths = watch_paths or get_default_watch_paths()
        self.recursive = recursive
        self.force_polling = force_polling
        self.backend = "disabled"
        self._lock = threading.Lock()
        self._counts = defaultdict(int)
        self._observer = None
        self._snapshot = {}
        self._started = False
        self._last_error = ""
        self._last_scan_time = None

    def start(self):
        if self._started:
            return self

        if not self.watch_paths:
            self.backend = "disabled"
            self._started = True
            return self

        if not self.force_polling and Observer is not None and FileSystemEventHandler is not None:
            try:
                observer = Observer()
                handler = _WatchdogCounter(self)
                for path in self.watch_paths:
                    observer.schedule(handler, path, recursive=self.recursive)
                observer.start()
                self._observer = observer
                self.backend = "watchdog"
                self._started = True
                return self
            except Exception as exc:
                self._last_error = str(exc)
                try:
                    if self._observer:
                        self._observer.stop()
                        self._observer.join(timeout=1)
                except Exception:
                    pass
                self._observer = None

        self._snapshot = self._build_snapshot()
        self.backend = "polling"
        self._started = True
        return self

    def stop(self):
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2)
            except Exception:
                pass
        self._observer = None
        self._started = False

    def reset_counts(self):
        with self._lock:
            self._counts = defaultdict(int)

    def _increment(self, event_name, amount=1):
        with self._lock:
            self._counts[event_name] += amount

    def _build_snapshot(self):
        snapshot = {}
        for path in self.watch_paths:
            for root, dirs, files in os.walk(path):
                dirs[:] = [name for name in dirs if not self._should_skip(os.path.join(root, name))]
                for name in files:
                    file_path = os.path.join(root, name)
                    if self._should_skip(file_path):
                        continue
                    try:
                        stat_result = os.stat(file_path)
                        snapshot[file_path] = (stat_result.st_mtime_ns, stat_result.st_size)
                    except OSError:
                        continue
        self._last_scan_time = time.time()
        return snapshot

    def _poll_once(self):
        current = self._build_snapshot()
        previous = self._snapshot

        created = set(current) - set(previous)
        deleted = set(previous) - set(current)
        modified = {
            path for path in set(current).intersection(previous)
            if current[path] != previous[path]
        }

        if created:
            self._increment("created", len(created))
        if deleted:
            self._increment("deleted", len(deleted))
        if modified:
            self._increment("modified", len(modified))

        self._snapshot = current

    @staticmethod
    def _should_skip(path):
        name = os.path.basename(path).lower()
        return name.endswith((".tmp", ".temp", ".part", ".ds_store"))

    def get_counts(self, reset=False):
        if not self._started:
            self.start()

        if self.backend == "polling":
            self._poll_once()

        with self._lock:
            counts = {
                "created": int(self._counts.get("created", 0)),
                "modified": int(self._counts.get("modified", 0)),
                "deleted": int(self._counts.get("deleted", 0)),
            }
            if reset:
                self._counts = defaultdict(int)

        counts["backend"] = self.backend
        counts["watch_paths"] = list(self.watch_paths)
        counts["last_error"] = self._last_error
        counts["last_scan_time"] = self._last_scan_time
        return counts
