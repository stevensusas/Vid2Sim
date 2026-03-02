#!/usr/bin/env python3
"""Batch runner: run the Vid2Sim pipeline on all (or a subset of) cases.

Usage:
    conda activate vid2sim
    cd Vid2Sim
    python run_pipeline_batch.py                          # all cases
    python run_pipeline_batch.py --cases double_stretch_zebra single_lift_sloth
    python run_pipeline_batch.py --skip double_stretch_zebra  # skip already-done cases
"""

import argparse
import os
import subprocess
import sys
import time
import traceback

ALL_CASES = [
    "double_lift_cloth_1",
    "double_lift_cloth_3",
    "double_lift_sloth",
    "double_lift_zebra",
    "double_stretch_sloth",
    "double_stretch_zebra",
    "rope_double_hand",
    "single_clift_cloth_1",
    "single_clift_cloth_3",
    "single_lift_cloth",
    "single_lift_cloth_1",
    "single_lift_cloth_3",
    "single_lift_cloth_4",
    "single_lift_dinosor",
    "single_lift_sloth",
    "single_lift_zebra",
    "single_push_sloth",
    "weird_package",
]

PYTHON = sys.executable  # same python that launched this script

def run_case(case: str, args) -> None:
    cmd = [
        PYTHON, "run_pipeline.py",
        "--config",           args.config,
        "--dataset_dir",      args.dataset_dir,
        "--output_dir",       args.output_dir,
        "--data_name",        case,
        "--ckpt_predictor",   args.ckpt_predictor,
        "--ckpt_lbs",         args.ckpt_lbs,
        "--ckpt_lgm",         args.ckpt_lgm,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases",          nargs="*", default=None,
                        help="Subset of cases to run (default: all)")
    parser.add_argument("--skip",           nargs="*", default=[],
                        help="Cases to skip")
    parser.add_argument("--config",         default="config/gso.yaml")
    parser.add_argument("--dataset_dir",    default="../PhysTwin/vid2sim_dataset")
    parser.add_argument("--output_dir",     default="outputs")
    parser.add_argument("--ckpt_predictor", default="checkpoints/ckpt_phys_predictor.pth")
    parser.add_argument("--ckpt_lbs",       default="checkpoints/ckpt_lbs_template.pth")
    parser.add_argument("--ckpt_lgm",       default="checkpoints/ckpt_lgm.safetensors")
    args = parser.parse_args()

    cases = args.cases if args.cases else ALL_CASES
    cases = [c for c in cases if c not in args.skip]

    results = {}
    t0 = time.time()

    for i, case in enumerate(cases):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(cases)}] {case}")
        print(f"{'='*60}")
        case_t0 = time.time()
        try:
            run_case(case, args)
            elapsed = time.time() - case_t0
            results[case] = f"OK ({elapsed/60:.1f} min)"
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - case_t0
            results[case] = f"FAILED ({elapsed/60:.1f} min)"
            print(f"[{case}] FAILED: {e}")

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE — {total/60:.1f} min total")
    print(f"{'='*60}")
    for case, status in results.items():
        mark = "✓" if status.startswith("OK") else "✗"
        print(f"  {mark} {case}: {status}")
