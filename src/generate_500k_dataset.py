"""
generate_500k_dataset.py
Generates a 1,000,000 sample behavioral dataset:
  - Collects 1,000 real normal snapshots from your laptop (~8 min)
  - Synthesizes 499,000 more normal samples from real distribution
  - Generates 500,000 attack samples (100,000 per variant x 5 variants)
  Total: 1,000,000 samples

Run from project root:
    python run_500k_dataset.py
"""

import os
import time
import psutil
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

SAVE_PATH       = "data/behavioral"
MODELS_DIR      = "models"
REAL_NORMAL     = 1000        # real snapshots to collect
TOTAL_NORMAL    = 500000      # total normal samples
TOTAL_ATTACK    = 500000      # total attack samples (100k per variant)
INTERVAL        = 0.5         # seconds between real snapshots
CHUNK_SIZE      = 100000      # save in chunks to avoid memory issues

os.makedirs(SAVE_PATH,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "cpu_percent",
    "memory_percent",
    "process_count",
    "high_cpu_process_count",
    "active_connections",
    "established_connections",
    "unique_remote_ports",
    "bytes_sent_rate",
    "bytes_recv_rate",
    "file_modified_count",
    "file_created_count",
    "file_deleted_count",
    "disk_write_rate",
    "new_process_count",
]

COUNT_FEATURES = {
    "process_count",
    "high_cpu_process_count",
    "active_connections",
    "established_connections",
    "unique_remote_ports",
    "file_modified_count",
    "file_created_count",
    "file_deleted_count",
    "new_process_count",
}

PERCENT_FEATURES = {"cpu_percent", "memory_percent"}

NORMAL_OVERLAP_RATE = 0.08
ATTACK_OVERLAP_RATE = 0.18
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

_prev_net  = None
_prev_disk = None
_prev_time = None
_prev_pids = set()


def collect_one():
    global _prev_net, _prev_disk, _prev_time, _prev_pids
    now  = time.time()
    snap = {}

    snap["cpu_percent"]    = psutil.cpu_percent(interval=0.1)
    snap["memory_percent"] = psutil.virtual_memory().percent

    try:
        procs = list(psutil.process_iter(["pid","cpu_percent"]))
        current_pids = set(p.info["pid"] for p in procs)
        snap["process_count"]          = len(procs)
        snap["high_cpu_process_count"] = sum(1 for p in procs
            if (p.info.get("cpu_percent") or 0) > 5)
        snap["new_process_count"]      = len(current_pids - _prev_pids)
        _prev_pids = current_pids
    except Exception:
        snap["process_count"] = 100
        snap["high_cpu_process_count"] = 2
        snap["new_process_count"] = 0

    try:
        net   = psutil.net_io_counters()
        conns = psutil.net_connections()
        snap["active_connections"]      = len(conns)
        snap["established_connections"] = sum(1 for c in conns if c.status=="ESTABLISHED")
        snap["unique_remote_ports"]     = len(set(c.raddr.port for c in conns if c.raddr))
        if _prev_net and _prev_time:
            dt = max(now - _prev_time, 0.1)
            snap["bytes_sent_rate"] = max(0,(net.bytes_sent-_prev_net.bytes_sent)/dt)
            snap["bytes_recv_rate"] = max(0,(net.bytes_recv-_prev_net.bytes_recv)/dt)
        else:
            snap["bytes_sent_rate"] = 0.0
            snap["bytes_recv_rate"] = 0.0
        _prev_net = net
    except Exception:
        snap["active_connections"] = 30
        snap["established_connections"] = 10
        snap["unique_remote_ports"] = 5
        snap["bytes_sent_rate"] = 0.0
        snap["bytes_recv_rate"] = 0.0

    try:
        disk = psutil.disk_io_counters()
        if _prev_disk and _prev_time:
            dt = max(now - _prev_time, 0.1)
            snap["disk_write_rate"] = max(0,(disk.write_bytes-_prev_disk.write_bytes)/dt)
        else:
            snap["disk_write_rate"] = 0.0
        _prev_disk = disk
    except Exception:
        snap["disk_write_rate"] = 0.0

    snap["file_modified_count"] = 0
    snap["file_created_count"]  = 0
    snap["file_deleted_count"]  = 0
    _prev_time = now
    return snap


def collect_real_normal(n=REAL_NORMAL):
    print(f"\n-- Collecting {n} real normal snapshots (~{int(n*INTERVAL/60)+1} min) --")
    print("  Use your laptop normally — browse, open apps, watch videos.\n")
    collect_one()
    time.sleep(1)

    rows = []
    for i in range(n):
        snap = collect_one()
        rows.append(snap)
        if (i+1) % 100 == 0:
            print(f"  [{i+1:4d}/{n}] CPU:{snap['cpu_percent']:5.1f}%  "
                  f"Mem:{snap['memory_percent']:5.1f}%  "
                  f"Procs:{snap['process_count']}  "
                  f"Conns:{snap['active_connections']}")
        time.sleep(INTERVAL)

    print(f"\n  Real collection done: {len(rows)} snapshots")
    return rows


def compute_stats(df):
    stats = {}
    for feat in FEATURE_NAMES:
        vals = df[feat].values
        stats[feat] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "p05":  float(np.percentile(vals, 5)),
            "p25":  float(np.percentile(vals, 25)),
            "p75":  float(np.percentile(vals, 75)),
            "p95":  float(np.percentile(vals, 95)),
        }
    return stats


def build_generation_profile(df):
    stats = compute_stats(df)
    cov = df[FEATURE_NAMES].cov().fillna(0).values.astype(np.float64)
    diag = np.diag([max(stats[feat]["std"], 1.0) ** 2 for feat in FEATURE_NAMES])
    cov = (0.80 * cov) + (0.20 * diag)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    stable_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return {
        "stats": stats,
        "covariance": stable_cov,
        "real_df": df[FEATURE_NAMES].reset_index(drop=True).copy(),
    }


def _sample_bootstrap_rows(profile, n):
    real_df = profile["real_df"]
    idx = np.random.randint(0, len(real_df), size=n)
    return real_df.iloc[idx].reset_index(drop=True).copy()


def _sample_correlated_rows(profile, n):
    means = np.array([profile["stats"][feat]["mean"] for feat in FEATURE_NAMES], dtype=np.float64)
    draws = np.random.multivariate_normal(
        mean=means,
        cov=profile["covariance"],
        size=n,
        check_valid="ignore",
    )
    return pd.DataFrame(draws, columns=FEATURE_NAMES)


def _add_measurement_noise(df, stats, scale=0.08):
    noisy = df.copy()
    for feat in FEATURE_NAMES:
        std = max(stats[feat]["std"], 0.01)
        noisy[feat] += np.random.normal(0, std * scale, len(noisy))
    return noisy


def _inject_benign_bursts(df, burst_rate=0.08):
    if len(df) == 0:
        return df

    noisy = df.copy()
    mask = np.random.rand(len(noisy)) < burst_rate
    if not mask.any():
        return noisy

    count = int(mask.sum())
    noisy.loc[mask, "cpu_percent"] *= np.random.uniform(1.05, 1.65, count)
    noisy.loc[mask, "memory_percent"] *= np.random.uniform(1.01, 1.20, count)
    noisy.loc[mask, "disk_write_rate"] *= np.random.uniform(1.20, 3.50, count)
    noisy.loc[mask, "bytes_sent_rate"] *= np.random.uniform(1.10, 2.80, count)
    noisy.loc[mask, "bytes_recv_rate"] *= np.random.uniform(1.05, 2.50, count)
    noisy.loc[mask, "active_connections"] += np.random.randint(0, 18, count)
    noisy.loc[mask, "established_connections"] += np.random.randint(0, 10, count)
    noisy.loc[mask, "file_modified_count"] += np.random.poisson(4, count)
    noisy.loc[mask, "file_created_count"] += np.random.poisson(2, count)
    noisy.loc[mask, "new_process_count"] += np.random.randint(0, 5, count)
    return noisy


def _severity_vector(n, low=0.35, high=1.0, overlap_ratio=0.20):
    severity = low + (high - low) * np.random.beta(2.0, 1.8, n)
    overlap_mask = np.random.rand(n) < overlap_ratio
    if overlap_mask.any():
        severity[overlap_mask] *= np.random.uniform(0.25, 0.65, int(overlap_mask.sum()))
    return severity


def _blend_feature(df, feature, low, high, severity, touch_prob=1.0, noise_ratio=0.08):
    blended = df.copy()
    current = blended[feature].to_numpy(dtype=np.float64)
    target = np.random.uniform(low, high, len(blended))
    active = np.random.rand(len(blended)) < touch_prob
    step = severity * np.random.uniform(0.55, 1.0, len(blended))
    current[active] = current[active] + step[active] * (target[active] - current[active])
    current += np.random.normal(0, max(high - low, 1.0) * noise_ratio, len(blended))
    blended[feature] = current
    return blended


def _finalize_rows(df, stats, clip_mode="attack"):
    clean = df.copy()
    upper_scale = 1.35 if clip_mode == "normal" else 4.0
    fallback_scale = 1.20 if clip_mode == "normal" else 3.0
    std_scale = 4 if clip_mode == "normal" else 12

    for feat in FEATURE_NAMES:
        stat = stats[feat]
        if feat in PERCENT_FEATURES:
            high = 100.0
        else:
            high = max(
                stat["p95"] * upper_scale,
                stat["max"] * fallback_scale,
                stat["mean"] + std_scale * max(stat["std"], 1.0),
                1.0,
            )
        clean[feat] = np.clip(clean[feat], 0, high)

    process_floor = np.maximum(clean["high_cpu_process_count"], clean["new_process_count"])
    process_floor += np.random.randint(1, 25, len(clean))
    clean["process_count"] = np.maximum(clean["process_count"], process_floor)

    clean["active_connections"] = np.maximum(clean["active_connections"], clean["established_connections"])
    clean["established_connections"] = np.minimum(clean["established_connections"], clean["active_connections"])
    clean["unique_remote_ports"] = np.minimum(
        clean["unique_remote_ports"],
        np.maximum(clean["active_connections"], 1),
    )
    clean["high_cpu_process_count"] = np.minimum(clean["high_cpu_process_count"], clean["process_count"])
    clean["new_process_count"] = np.minimum(clean["new_process_count"], clean["process_count"])

    file_ops = (
        clean["file_modified_count"]
        + clean["file_created_count"]
        + clean["file_deleted_count"]
    )
    min_disk = file_ops * np.random.uniform(2e4, 1.2e5, len(clean))
    clean["disk_write_rate"] = np.where(
        file_ops > 0,
        np.maximum(clean["disk_write_rate"], min_disk),
        clean["disk_write_rate"],
    )
    clean["bytes_recv_rate"] = np.maximum(
        clean["bytes_recv_rate"],
        clean["bytes_sent_rate"] * np.random.uniform(0.10, 0.75, len(clean)),
    )

    for feat in COUNT_FEATURES:
        clean[feat] = np.round(np.clip(clean[feat], 0, None)).astype(float)

    return clean[FEATURE_NAMES]


def _generate_base_rows(profile, n, include_bursts=True):
    boot = _sample_bootstrap_rows(profile, n)
    corr = _sample_correlated_rows(profile, n)
    weights = np.random.uniform(0.35, 0.75, size=(n, 1))
    blended = (boot.values * weights) + (corr.values * (1 - weights))
    df = pd.DataFrame(blended, columns=FEATURE_NAMES)
    df = _add_measurement_noise(df, profile["stats"], scale=0.10)
    if include_bursts:
        df = _inject_benign_bursts(df, burst_rate=0.07)
    return _finalize_rows(df, profile["stats"], clip_mode="normal")


def synthesize_normal(profile, n, chunk_size=CHUNK_SIZE):
    print(f"\n-- Synthesizing {n:,} normal samples --")
    all_chunks = []
    generated  = 0
    chunk_num  = 0

    while generated < n:
        curr = min(chunk_size, n - generated)
        chunk_df = _generate_base_rows(profile, curr, include_bursts=True)
        all_chunks.append(chunk_df)
        generated += curr
        chunk_num += 1
        if chunk_num % 2 == 0:
            print(f"  Generated {generated:,}/{n:,} normal samples...")

    print(f"  Normal synthesis complete: {generated:,} samples")
    return pd.concat(all_chunks, ignore_index=True)


def gen_fast_encryption(profile, n):
    """WannaCry-style: strong encryption burst with partial overlap."""
    df = _generate_base_rows(profile, n, include_bursts=True)
    severity = _severity_vector(n, low=0.55, high=1.0, overlap_ratio=0.20)
    df = _blend_feature(df, "cpu_percent", 55, 99, severity, touch_prob=0.95, noise_ratio=0.06)
    df = _blend_feature(df, "memory_percent", 55, 98, severity, touch_prob=0.85, noise_ratio=0.05)
    df = _blend_feature(df, "high_cpu_process_count", 3, 20, severity, touch_prob=0.95, noise_ratio=0.08)
    df = _blend_feature(df, "new_process_count", 4, 30, severity, touch_prob=0.90, noise_ratio=0.08)
    df = _blend_feature(df, "file_modified_count", 20, 400, severity, touch_prob=0.98, noise_ratio=0.10)
    df = _blend_feature(df, "file_created_count", 5, 200, severity, touch_prob=0.92, noise_ratio=0.10)
    df = _blend_feature(df, "file_deleted_count", 0, 250, severity, touch_prob=0.90, noise_ratio=0.10)
    df = _blend_feature(df, "disk_write_rate", 2e6, 100e6, severity, touch_prob=0.98, noise_ratio=0.08)
    df = _blend_feature(df, "bytes_sent_rate", 2e4, 6e5, severity, touch_prob=0.75, noise_ratio=0.06)
    return _finalize_rows(df, profile["stats"], clip_mode="attack")


def gen_slow_encryption(profile, n):
    """Low-and-slow encryption that intentionally overlaps normal behavior."""
    df = _generate_base_rows(profile, n, include_bursts=True)
    severity = _severity_vector(n, low=0.25, high=0.75, overlap_ratio=0.35)
    df = _blend_feature(df, "cpu_percent", 25, 68, severity, touch_prob=0.75, noise_ratio=0.09)
    df = _blend_feature(df, "memory_percent", 40, 82, severity, touch_prob=0.65, noise_ratio=0.07)
    df = _blend_feature(df, "high_cpu_process_count", 1, 8, severity, touch_prob=0.70, noise_ratio=0.10)
    df = _blend_feature(df, "new_process_count", 1, 10, severity, touch_prob=0.70, noise_ratio=0.10)
    df = _blend_feature(df, "file_modified_count", 2, 45, severity, touch_prob=0.85, noise_ratio=0.12)
    df = _blend_feature(df, "file_created_count", 0, 18, severity, touch_prob=0.65, noise_ratio=0.12)
    df = _blend_feature(df, "file_deleted_count", 0, 22, severity, touch_prob=0.60, noise_ratio=0.12)
    df = _blend_feature(df, "disk_write_rate", 0.2e6, 8e6, severity, touch_prob=0.85, noise_ratio=0.10)
    df = _blend_feature(df, "bytes_sent_rate", 2e3, 8e4, severity, touch_prob=0.70, noise_ratio=0.08)
    return _finalize_rows(df, profile["stats"], clip_mode="attack")


def gen_fileless_attack(profile, n):
    """Memory-resident: strong memory/network pattern with low disk pressure."""
    df = _generate_base_rows(profile, n, include_bursts=True)
    severity = _severity_vector(n, low=0.35, high=0.90, overlap_ratio=0.25)
    df = _blend_feature(df, "cpu_percent", 35, 88, severity, touch_prob=0.80, noise_ratio=0.08)
    df = _blend_feature(df, "memory_percent", 65, 99, severity, touch_prob=0.98, noise_ratio=0.05)
    df = _blend_feature(df, "active_connections", 20, 150, severity, touch_prob=0.95, noise_ratio=0.08)
    df = _blend_feature(df, "established_connections", 10, 100, severity, touch_prob=0.90, noise_ratio=0.08)
    df = _blend_feature(df, "unique_remote_ports", 5, 60, severity, touch_prob=0.90, noise_ratio=0.08)
    df = _blend_feature(df, "bytes_sent_rate", 5e4, 1e6, severity, touch_prob=0.92, noise_ratio=0.08)
    df = _blend_feature(df, "bytes_recv_rate", 2e4, 7e5, severity, touch_prob=0.82, noise_ratio=0.08)
    df = _blend_feature(df, "new_process_count", 2, 18, severity, touch_prob=0.75, noise_ratio=0.10)
    df = _blend_feature(df, "disk_write_rate", 0.05e6, 4e6, severity * 0.60, touch_prob=0.60, noise_ratio=0.12)
    df = _blend_feature(df, "file_modified_count", 0, 20, severity * 0.55, touch_prob=0.50, noise_ratio=0.12)
    return _finalize_rows(df, profile["stats"], clip_mode="attack")


def gen_polymorphic_attack(profile, n):
    """Phase-changing attack with mixed behaviors and overlapping transitions."""
    df = _generate_base_rows(profile, n, include_bursts=True)
    severity = _severity_vector(n, low=0.30, high=0.95, overlap_ratio=0.25)
    phases = np.arange(n) % 4

    for phase in range(4):
        mask = phases == phase
        if not mask.any():
            continue

        phase_df = df.loc[mask, FEATURE_NAMES].copy()
        phase_severity = severity[mask]

        if phase == 0:
            phase_df = _blend_feature(phase_df, "cpu_percent", 45, 99, phase_severity, 0.95, 0.07)
            phase_df = _blend_feature(phase_df, "file_modified_count", 10, 300, phase_severity, 0.95, 0.10)
            phase_df = _blend_feature(phase_df, "disk_write_rate", 1e6, 60e6, phase_severity, 0.95, 0.08)
        elif phase == 1:
            phase_df = _blend_feature(phase_df, "active_connections", 15, 120, phase_severity, 0.90, 0.08)
            phase_df = _blend_feature(phase_df, "unique_remote_ports", 5, 70, phase_severity, 0.90, 0.08)
            phase_df = _blend_feature(phase_df, "bytes_sent_rate", 5e4, 6e5, phase_severity, 0.90, 0.08)
        elif phase == 2:
            phase_df = _blend_feature(phase_df, "new_process_count", 4, 40, phase_severity, 0.95, 0.10)
            phase_df = _blend_feature(phase_df, "high_cpu_process_count", 2, 18, phase_severity, 0.90, 0.10)
            phase_df = _blend_feature(phase_df, "memory_percent", 55, 92, phase_severity, 0.85, 0.06)
        else:
            phase_df = _blend_feature(phase_df, "cpu_percent", 18, 55, phase_severity * 0.80, 0.70, 0.08)
            phase_df = _blend_feature(phase_df, "file_modified_count", 0, 40, phase_severity * 0.65, 0.65, 0.10)
            phase_df = _blend_feature(phase_df, "bytes_sent_rate", 0, 1.5e5, phase_severity * 0.60, 0.60, 0.10)

        df.loc[mask, FEATURE_NAMES] = phase_df[FEATURE_NAMES].values

    return _finalize_rows(df, profile["stats"], clip_mode="attack")


def gen_network_heavy_attack(profile, n):
    """Exfiltration-heavy attack with softer overlap into normal traffic."""
    df = _generate_base_rows(profile, n, include_bursts=True)
    severity = _severity_vector(n, low=0.40, high=0.95, overlap_ratio=0.22)
    df = _blend_feature(df, "active_connections", 25, 200, severity, touch_prob=0.98, noise_ratio=0.08)
    df = _blend_feature(df, "established_connections", 10, 120, severity, touch_prob=0.95, noise_ratio=0.08)
    df = _blend_feature(df, "unique_remote_ports", 10, 80, severity, touch_prob=0.95, noise_ratio=0.08)
    df = _blend_feature(df, "bytes_sent_rate", 8e4, 5e6, severity, touch_prob=0.98, noise_ratio=0.08)
    df = _blend_feature(df, "bytes_recv_rate", 5e4, 2e6, severity, touch_prob=0.95, noise_ratio=0.08)
    df = _blend_feature(df, "cpu_percent", 25, 82, severity, touch_prob=0.80, noise_ratio=0.07)
    df = _blend_feature(df, "new_process_count", 2, 18, severity, touch_prob=0.70, noise_ratio=0.10)
    df = _blend_feature(df, "file_modified_count", 5, 120, severity * 0.85, touch_prob=0.80, noise_ratio=0.10)
    df = _blend_feature(df, "disk_write_rate", 0.5e6, 25e6, severity * 0.90, touch_prob=0.85, noise_ratio=0.08)
    return _finalize_rows(df, profile["stats"], clip_mode="attack")


def introduce_class_overlap(normal_df, attack_df, profile):
    normal_idx = normal_df.index[np.random.rand(len(normal_df)) < NORMAL_OVERLAP_RATE]
    attack_idx = attack_df.index[np.random.rand(len(attack_df)) < ATTACK_OVERLAP_RATE]

    if len(normal_idx) > 0:
        borderline = gen_slow_encryption(profile, len(normal_idx))
        normal_weights = np.random.uniform(0.65, 0.85, size=(len(normal_idx), 1))
        blended = (
            normal_df.loc[normal_idx, FEATURE_NAMES].to_numpy(dtype=np.float64) * normal_weights
            + borderline[FEATURE_NAMES].to_numpy(dtype=np.float64) * (1 - normal_weights)
        )
        normal_df.loc[normal_idx, FEATURE_NAMES] = _finalize_rows(
            pd.DataFrame(blended, columns=FEATURE_NAMES),
            profile["stats"],
            clip_mode="normal",
        ).values

    if len(attack_idx) > 0:
        baseline = _generate_base_rows(profile, len(attack_idx), include_bursts=True)
        attack_weights = np.random.uniform(0.55, 0.80, size=(len(attack_idx), 1))
        blended = (
            attack_df.loc[attack_idx, FEATURE_NAMES].to_numpy(dtype=np.float64) * attack_weights
            + baseline[FEATURE_NAMES].to_numpy(dtype=np.float64) * (1 - attack_weights)
        )
        attack_df.loc[attack_idx, FEATURE_NAMES] = _finalize_rows(
            pd.DataFrame(blended, columns=FEATURE_NAMES),
            profile["stats"],
            clip_mode="attack",
        ).values

    return normal_df, attack_df, len(normal_idx), len(attack_idx)


def build_1m_dataset(real_normal_rows):
    print(f"\n-- Building 1,000,000 sample dataset ----------------")

    real_df = pd.DataFrame(real_normal_rows)[FEATURE_NAMES]
    generation_profile = build_generation_profile(real_df)
    real_stats = generation_profile["stats"]

    with open(os.path.join(MODELS_DIR, "normal_stats.pkl"), "wb") as f:
        pickle.dump(real_stats, f)
    with open(os.path.join(MODELS_DIR, "behavioral_feature_names.pkl"), "wb") as f:
        pickle.dump(FEATURE_NAMES, f)

    # Synthesize normal samples
    synth_normal = synthesize_normal(generation_profile, TOTAL_NORMAL - len(real_normal_rows))
    normal_df    = pd.concat([real_df, synth_normal], ignore_index=True)
    normal_df["label"] = 0
    print(f"  Normal samples total : {len(normal_df):,}")

    # Generate 5 attack variants × 100K each = 500K
    n_per = TOTAL_ATTACK // 5
    print(f"\n-- Generating {TOTAL_ATTACK:,} attack samples ({n_per:,} per variant) --")

    variants = [
        ("Fast encryption (WannaCry)",   gen_fast_encryption(generation_profile, n_per)),
        ("Slow encryption (Evasion)",    gen_slow_encryption(generation_profile, n_per)),
        ("Fileless attack (Memory)",     gen_fileless_attack(generation_profile, n_per)),
        ("Polymorphic (Phase-changing)", gen_polymorphic_attack(generation_profile, n_per)),
        ("Network-heavy (Ryuk/REvil)",   gen_network_heavy_attack(generation_profile, n_per)),
    ]

    attack_chunks = []
    for name, df_v in variants:
        print(f"  {name}: {len(df_v):,} samples")
        df_v["label"] = 1
        attack_chunks.append(df_v)
    attack_df = pd.concat(attack_chunks, ignore_index=True)

    print("\n-- Injecting overlap and noisy edge cases ------------")
    normal_df, attack_df, normal_overlap, attack_overlap = introduce_class_overlap(
        normal_df,
        attack_df,
        generation_profile,
    )
    print(f"  Normal borderline samples : {normal_overlap:,}")
    print(f"  Attack overlap samples    : {attack_overlap:,}")

    # Combine and shuffle
    print(f"\n-- Combining and shuffling ---------------------------")
    full_df = pd.concat([normal_df, attack_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    full_df = full_df.fillna(0).replace([np.inf, -np.inf], 0)

    print(f"  Total samples : {len(full_df):,}")
    print(f"  Normal (0)    : {(full_df['label']==0).sum():,}")
    print(f"  Attack (1)    : {(full_df['label']==1).sum():,}")
    print(f"  Features      : {len(FEATURE_NAMES)}")

    # Save
    path = os.path.join(SAVE_PATH, "dataset_500k.csv")
    print(f"\n-- Saving dataset ------------------------------------")
    full_df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved -> {path}  ({size_mb:.1f} MB)")

    return full_df, real_stats


def main():
    print("=" * 58)
    print("  1,000,000 SAMPLE BEHAVIORAL DATASET GENERATOR")
    print("=" * 58)
    print(f"  Real normal samples  : {REAL_NORMAL:,}  (~8 min collection)")
    print(f"  Synthetic normal     : {TOTAL_NORMAL-REAL_NORMAL:,}")
    print(f"  Total normal         : {TOTAL_NORMAL:,}")
    print(f"  Attack variants      : 5 ransomware types")
    print(f"  Attack per variant   : {TOTAL_ATTACK//5:,}")
    print(f"  Total attack         : {TOTAL_ATTACK:,}")
    print(f"  GRAND TOTAL          : {TOTAL_NORMAL+TOTAL_ATTACK:,}")
    print("=" * 58)

    real_rows      = collect_real_normal(REAL_NORMAL)
    full_df, stats = build_1m_dataset(real_rows)

    print("\n" + "=" * 58)
    print("  1M dataset generation complete!")
    print("  Now run: python run_500k_training.py")
    print("=" * 58)
    return full_df


if __name__ == "__main__":
    main()
