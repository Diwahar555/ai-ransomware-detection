"""
retrain_from_live_data.py
=========================
Retrain behavioral models using YOUR real captured normal data from
logs/live_data.csv instead of relying on synthetic-only normal samples.

This dramatically reduces false positives because the models learn
exactly what "normal" looks like on THIS machine.

Run from project root:
    python retrain_from_live_data.py

Steps performed:
  1. Load and clean logs/live_data.csv  (your real normal snapshots)
  2. Synthesize additional normal samples from the real distribution
  3. Generate 500K attack samples (same 5 ransomware variants)
  4. Retrain all 5 models + meta-ensemble
  5. Save updated models to models/
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generate_500k_dataset import (
    FEATURE_NAMES,
    build_generation_profile,
    synthesize_normal,
    gen_fast_encryption,
    gen_slow_encryption,
    gen_fileless_attack,
    gen_polymorphic_attack,
    gen_network_heavy_attack,
    introduce_class_overlap,
    _finalize_rows,
    MODELS_DIR,
    SAVE_PATH,
    TOTAL_ATTACK,
    CHUNK_SIZE,
)
from src.train_500k_models import run_500k_training

# ── Config ────────────────────────────────────────────────────────────────────
LIVE_CSV      = "logs/live_data.csv"
TOTAL_NORMAL  = 500_000      # total normal samples to produce (real + synthetic)
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

os.makedirs(SAVE_PATH,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_clean_live_data(path: str) -> pd.DataFrame:
    """Load live_data.csv, keep only valid feature rows, deduplicate."""
    print(f"\n-- Loading real normal data from {path}")
    df = pd.read_csv(path)
    print(f"   Raw rows: {len(df):,}")

    # Keep only rows that have all required features
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"live_data.csv is missing columns: {missing}")

    df = df[FEATURE_NAMES].copy()

    # Drop rows where all numeric features are zero (sensor warm-up rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    all_zero_mask = (df[FEATURE_NAMES] == 0).all(axis=1)
    df = df[~all_zero_mask]

    # Clip extreme outliers: keep values within 99.5th percentile per feature
    for feat in FEATURE_NAMES:
        upper = df[feat].quantile(0.995)
        df[feat] = df[feat].clip(upper=upper)

    # Remove duplicate rows (exact same reading, e.g., logged twice)
    df = df.drop_duplicates()

    # Ensure non-negative counts
    for feat in FEATURE_NAMES:
        df[feat] = df[feat].clip(lower=0)

    print(f"   Clean rows kept: {len(df):,}")
    return df.reset_index(drop=True)


def build_dataset_from_live(real_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n-- Building training dataset from {len(real_df):,} real normal rows")

    generation_profile = build_generation_profile(real_df)
    real_stats = generation_profile["stats"]

    import pickle
    with open(os.path.join(MODELS_DIR, "normal_stats.pkl"), "wb") as f:
        pickle.dump(real_stats, f)
    with open(os.path.join(MODELS_DIR, "behavioral_feature_names.pkl"), "wb") as f:
        pickle.dump(FEATURE_NAMES, f)

    # Synthesize remaining normal samples to reach TOTAL_NORMAL
    remaining = max(TOTAL_NORMAL - len(real_df), 0)
    if remaining > 0:
        synth_normal = synthesize_normal(generation_profile, remaining, chunk_size=CHUNK_SIZE)
        normal_df = pd.concat([real_df, synth_normal], ignore_index=True)
    else:
        normal_df = real_df.copy()
    normal_df["label"] = 0
    print(f"   Normal samples total : {len(normal_df):,}")

    # Generate attack samples
    n_per = TOTAL_ATTACK // 5
    print(f"\n-- Generating {TOTAL_ATTACK:,} attack samples ({n_per:,} per variant)")
    variants = [
        ("Fast encryption (WannaCry)",   gen_fast_encryption(generation_profile, n_per)),
        ("Slow encryption (Evasion)",    gen_slow_encryption(generation_profile, n_per)),
        ("Fileless attack (Memory)",     gen_fileless_attack(generation_profile, n_per)),
        ("Polymorphic (Phase-changing)", gen_polymorphic_attack(generation_profile, n_per)),
        ("Network-heavy (Ryuk/REvil)",   gen_network_heavy_attack(generation_profile, n_per)),
    ]
    attack_chunks = []
    for name, df_v in variants:
        print(f"   {name}: {len(df_v):,} samples")
        df_v["label"] = 1
        attack_chunks.append(df_v)
    attack_df = pd.concat(attack_chunks, ignore_index=True)

    print("\n-- Injecting overlap and noisy edge cases")
    normal_df, attack_df, n_ov, a_ov = introduce_class_overlap(
        normal_df, attack_df, generation_profile
    )
    print(f"   Normal borderline : {n_ov:,}  |  Attack overlap : {a_ov:,}")

    full_df = pd.concat([normal_df, attack_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    full_df = full_df.fillna(0).replace([np.inf, -np.inf], 0)

    print(f"\n   Total samples : {len(full_df):,}")
    print(f"   Normal (0)    : {(full_df['label']==0).sum():,}")
    print(f"   Attack (1)    : {(full_df['label']==1).sum():,}")

    # Save dataset so run_500k_training can pick it up
    path = os.path.join(SAVE_PATH, "dataset_500k.csv")
    print(f"\n-- Saving dataset → {path}")
    full_df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1e6
    print(f"   Saved ({size_mb:.1f} MB)")

    return full_df


def main():
    print("=" * 60)
    print("  RETRAIN FROM LIVE DATA")
    print("  Uses your real captured normal behavior in live_data.csv")
    print("  to build a machine-specific model with far fewer FPs.")
    print("=" * 60)

    if not os.path.exists(LIVE_CSV):
        print(f"\n❌ {LIVE_CSV} not found. Run the dashboard first to capture data.")
        sys.exit(1)

    real_df = load_and_clean_live_data(LIVE_CSV)

    if len(real_df) < 100:
        print(f"\n⚠️  Only {len(real_df)} usable rows found. Capture more data before retraining.")
        sys.exit(1)

    print(f"\n✅ Using {len(real_df):,} real normal snapshots as training baseline.")

    build_dataset_from_live(real_df)

    print("\n-- Training models on new dataset (this will take several minutes)...")
    run_500k_training()

    print("\n" + "=" * 60)
    print("  ✅ Retraining complete!")
    print("  Models saved to models/")
    print("  Restart the dashboard to use the updated models.")
    print("=" * 60)


if __name__ == "__main__":
    main()
