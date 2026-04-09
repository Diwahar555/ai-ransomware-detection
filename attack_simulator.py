"""
attack_simulator.py
Simulates ransomware-like behavior SAFELY for data collection.
- Creates and modifies many files rapidly (like encryption)
- High CPU load (like encryption processing)
- All files are created in a TEMP folder and DELETED automatically
Run: python attack_simulator.py
"""

import os
import shutil
import time
import threading

DURATION = 180  # seconds to simulate (3 minutes)

# Try Desktop first, then Documents, then Home, then Current Directory
def get_safe_base_folder():
    paths_to_try = [
        os.path.join(os.path.expanduser("~"), "Documents", "attack_test_folder"),
        os.path.join(os.path.expanduser("~"), "Desktop", "attack_test_folder"),
        os.path.join(os.path.expanduser("~"), "attack_test_folder"),
        os.path.join(os.getcwd(), "attack_test_folder")
    ]
    for p in paths_to_try:
        try:
            os.makedirs(p, exist_ok=True)
            # Test write access
            test_file = os.path.join(p, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return p
        except Exception:
            continue
    return os.path.join(os.getcwd(), "attack_test_folder") # Absolute fallback

BASE_FOLDER = get_safe_base_folder()

def mass_file_operations():
    """Simulates rapid file creation and modification like ransomware encryption."""
    end_time = time.time() + DURATION
    batch = 0

    print(f"  [FILE] Creating/modifying files in: {BASE_FOLDER}")
    while time.time() < end_time:
        # Create 60 files per cycle (doubled to ensure watchdog picks them up)
        for j in range(60):
            path = os.path.join(BASE_FOLDER, f"enc_batch{batch}_file{j}.enc")
            with open(path, "wb") as f:
                f.write(os.urandom(1024 * 10))  # 10KB random data

        # Modify them (like ransomware re-writing encrypted content)
        for j in range(60):
            path = os.path.join(BASE_FOLDER, f"enc_batch{batch}_file{j}.enc")
            if os.path.exists(path):
                with open(path, "ab") as f:
                    f.write(os.urandom(512))

        batch += 1
        elapsed = int(time.time() - (end_time - DURATION))
        print(f"  [FILE] Batch {batch} done | {elapsed}s / {DURATION}s elapsed | {batch*60} files created")
        time.sleep(0.3)  # faster batches — more events per Streamlit refresh cycle


    # Cleanup safely
    shutil.rmtree(BASE_FOLDER, ignore_errors=True)
    print("  [FILE] Done. Temp folder cleaned up.")


def cpu_stress():
    """Simulates high CPU usage like ransomware encryption processing."""
    end_time = time.time() + DURATION
    print("  [CPU]  High CPU stress started...")
    while time.time() < end_time:
        _ = [x ** 2 for x in range(50000)]
    print("  [CPU]  Done.")


def main():
    print("=" * 50)
    print("  ATTACK SIMULATOR — Safe ransomware behavior simulation")
    print(f"  Duration : {DURATION} seconds")
    print(f"  Folder   : {BASE_FOLDER}  (auto-deleted after)")
    print("=" * 50)
    print("  Starting in 3 seconds...")
    time.sleep(3)

    t1 = threading.Thread(target=mass_file_operations)
    t2 = threading.Thread(target=cpu_stress)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("\n" + "=" * 50)
    print("  Simulation COMPLETE. Folder cleaned up safely.")
    print("=" * 50)


if __name__ == "__main__":
    main()
