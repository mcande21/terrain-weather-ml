"""Train the per-band 12-model pipeline on the full 1.3M row dataset."""
import sys
import time
import os

os.environ["OMP_NUM_THREADS"] = "4"

sys.path.insert(0, "/Users/cooperanderson/work/personal/code/colorado-avalanche-ml/src")

import duckdb
import numpy as np
import pandas as pd

print("Loading data from DuckDB...", flush=True)
t0 = time.time()
db = duckdb.connect("/Users/cooperanderson/work/personal/code/colorado-avalanche-ml/data/avalanche.duckdb", read_only=True)
df = db.execute("SELECT * FROM training_matrix").fetchdf()
db.close()
print(f"Loaded {len(df)} rows, {len(df.columns)} columns in {time.time()-t0:.1f}s", flush=True)
print(f"Date range: {df['date'].min()} to {df['date'].max()}", flush=True)
print(f"Elevation bands: {df['elevation_band'].unique().tolist()}", flush=True)
print(flush=True)

from avalanche_ml.models.per_band_pipeline import PerBandPipeline

print("Training per-band pipeline (12 models)...", flush=True)
print("  Stage 1: 3 problem types x 3 bands x 3 configs = 27 RF classifiers", flush=True)
print("  Stage 2: 3 bands = 3 RF classifiers", flush=True)
print(flush=True)

pipeline = PerBandPipeline()
t1 = time.time()
results = pipeline.train(df)
train_time = time.time() - t1
print(f"\nTraining complete in {train_time:.1f}s ({train_time/60:.1f} min)", flush=True)
print(flush=True)

SCHWARTZREICH = {
    "below_treeline": 0.544,
    "near_treeline": 0.525,
    "above_treeline": 0.508,
}

print("=" * 70, flush=True)
print("RESULTS: Per-Band Stage 2 (Danger Level Prediction)", flush=True)
print("=" * 70, flush=True)
print(flush=True)

for band in ["above_treeline", "near_treeline", "below_treeline"]:
    if band not in results:
        print(f"  {band}: SKIPPED (insufficient data)", flush=True)
        continue

    r = results[band]
    s2_test = r["stage2_test_metrics"]
    s2_val = r["stage2_val_metrics"]

    print(f"--- {band} ---", flush=True)
    print(f"  Split sizes: {r['split_sizes']}", flush=True)
    print(f"  Val  macro-F1: {s2_val['macro_f1']:.3f}", flush=True)
    print(f"  Test macro-F1: {s2_test['macro_f1']:.3f}", flush=True)
    print(f"  Schwartzreich: {SCHWARTZREICH[band]:.3f}", flush=True)
    delta = s2_test['macro_f1'] - SCHWARTZREICH[band]
    print(f"  Delta:         {delta:+.3f}", flush=True)
    print(f"  Per-class F1:  {s2_test.get('per_class_f1', {})}", flush=True)
    print(f"  FNR:           {s2_test.get('false_negative_rate', 'N/A'):.3f}", flush=True)
    print(f"  High danger:   {s2_test.get('high_danger_detection_rate', 'N/A'):.3f}", flush=True)
    print(flush=True)

print("=" * 70, flush=True)
print("RESULTS: Per-Band Stage 1 (Problem Type Prediction)", flush=True)
print("=" * 70, flush=True)
print(flush=True)

for band in ["above_treeline", "near_treeline", "below_treeline"]:
    if band not in results:
        continue

    r = results[band]
    s1 = r["stage1_eval_metrics"]
    print(f"--- {band} ---", flush=True)
    for pt in ["persistent_slab", "storm_slab", "loose_wet"]:
        if pt in s1:
            m = s1[pt]
            print(f"  {pt}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}", flush=True)
    ens = r["stage1_ensembles"]
    for pt in ["persistent_slab", "storm_slab", "loose_wet"]:
        if pt in ens:
            e = ens[pt]
            print(f"    ensemble: {e['n_configs_kept']}/{e['n_configs_trained']} configs, "
                  f"top scores: {[f'{s:.3f}' for s in e['scores']]}", flush=True)
    print(flush=True)

print("=" * 70, flush=True)
print("COMPARISON TABLE: Our Results vs Schwartzreich 2026", flush=True)
print("=" * 70, flush=True)
print(flush=True)
print(f"{'Band':<20} {'Ours (test)':<15} {'Schwartz.':<15} {'Delta':<10}", flush=True)
print("-" * 60, flush=True)
for band in ["above_treeline", "near_treeline", "below_treeline"]:
    if band in results:
        ours = results[band]["stage2_test_metrics"]["macro_f1"]
        theirs = SCHWARTZREICH[band]
        print(f"{band:<20} {ours:<15.3f} {theirs:<15.3f} {ours-theirs:+.3f}", flush=True)
print(flush=True)

# Feature importance
print("=" * 70, flush=True)
print("FEATURE IMPORTANCE: Stage-1 prediction rank in Stage-2", flush=True)
print("=" * 70, flush=True)
print(flush=True)
for band in ["above_treeline", "near_treeline", "below_treeline"]:
    if band not in results:
        continue
    clf = pipeline.stage2_models_[band]
    feat_names = pipeline.s2_feature_names_[band]
    importances = clf.feature_importances_
    pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)

    print(f"--- {band} Top 15 ---", flush=True)
    for name, imp in pairs[:15]:
        marker = " *** S1-PRED" if "_prob" in name else ""
        print(f"  {name:<35} {imp:.4f}{marker}", flush=True)

    prob_ranks = [i+1 for i, (n, _) in enumerate(pairs) if "_prob" in n]
    print(f"  Stage-1 prob features rank: {prob_ranks}", flush=True)
    print(flush=True)

# Save model
from pathlib import Path

save_dir = Path("/Users/cooperanderson/work/personal/code/colorado-avalanche-ml/models/per_band_v1")
print(f"Saving pipeline to {save_dir}...", flush=True)
pipeline.save(save_dir)
print("Saved.", flush=True)
print(flush=True)

print("=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"Total models: 12 (9 Stage-1 ensembles [3 configs each] + 3 Stage-2)", flush=True)
print(f"Training time: {train_time:.1f}s ({train_time/60:.1f} min)", flush=True)
print(f"Dataset: {len(df)} rows, {len(df.columns)} columns", flush=True)
print(f"Date range: {df['date'].min()} to {df['date'].max()}", flush=True)
