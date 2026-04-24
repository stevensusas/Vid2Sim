"""Standalone: render material_field.gif for a case that already finished.

Loads the trained MaterialMLP state dict and the cubature points from a completed
Vid2Sim run, skipping Stage I/II training entirely. Much faster than rerunning
the full pipeline.

Usage:
    python render_material_field.py \\
        --config config/gso.yaml \\
        --dataset_dir ../PhysTwin/vid2sim_dataset_interaction \\
        --output_dir outputs_interaction_mlp \\
        --data_name double_stretch_zebra_50
"""
import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulators.lbs_simulator import LBSSimulator
from run_pipeline import save_material_field_gif, _merge_per_case_sim_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/gso.yaml")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_name", type=str, required=True)
    p.add_argument("--out", type=str, default=None,
                   help="Output gif path (default: <output_dir>/<data_name>/material_field.gif)")
    args = p.parse_args()

    sim_args = OmegaConf.load(args.config)
    sim_args = _merge_per_case_sim_config(sim_args, args.dataset_dir, args.data_name)
    sim_args.tag = "best"
    # Dummy values — overridden by MaterialMLP state-dict load, but set_material()
    # still reads args.yms/prs for bias-init logging.
    sim_args.yms = 1e5
    sim_args.prs = 0.4

    simulator = LBSSimulator(sim_args, args.dataset_dir, args.output_dir, args.data_name)
    simulator.set_material()

    mat_path = os.path.join(args.output_dir, args.data_name, "models", "material_mlp_best.pth")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"No trained MaterialMLP at {mat_path}")
    simulator.material_mlp.load_state_dict(torch.load(mat_path, map_location=simulator.device))
    print(f"Loaded MaterialMLP from {mat_path}")

    out_path = args.out or os.path.join(args.output_dir, args.data_name, "material_field.gif")
    save_material_field_gif(simulator, out_path)


if __name__ == "__main__":
    main()
