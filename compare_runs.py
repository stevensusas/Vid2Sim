"""Side-by-side comparison visuals for baseline vs MaterialMLP+GripMLP runs.

For each case in CASES:
  1. overlay_compare.gif           — side-by-side overlay.gif
  2. tracking_v2_compare.gif       — side-by-side tracking_overlay_v2.gif
  3. optimization_compare.png      — joint PSNR/SSIM/Track/E/nu curves

Writes into {out_dir}/{case}_{name}.
"""
import argparse
import json
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = "/home/steven/project-steven/Vid2Sim/outputs_interaction"
TREAT = "/home/steven/project-steven/Vid2Sim/outputs_interaction_mlp"

DEFAULT_CASES = [
    "double_lift_cloth_1_50",
    "double_stretch_zebra_50",
    "single_lift_cloth_50",
    "single_lift_dinosor_50",
    "single_lift_sloth_50",
    "single_push_sloth_50",
]


def _font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _best_metrics(record_path):
    with open(record_path) as f:
        rec = json.load(f)
    tl = rec.get("test_list", [])
    if not tl:
        return None
    best = max(tl, key=lambda r: r.get("psnr", 0))
    return best


def make_side_by_side_gif(base_gif, treat_gif, out_path, *, left_label, right_label,
                           top_banner_text, fps=8, crop_left_half=False):
    a = imageio.mimread(base_gif, memtest=False)
    b = imageio.mimread(treat_gif, memtest=False)
    n = min(len(a), len(b))
    if crop_left_half:
        # tracking_overlay_*.gif stacks GT (left) + recon (right); keep only the GT
        # panel so the side-by-side doesn't double the doubling. Legend row is also
        # below the panels at H_img+legend_h — we keep everything; just halve width.
        half_w = a[0].shape[1] // 2
        a = [f[:, :half_w] for f in a]
        b = [f[:, :half_w] for f in b]
    H, W = a[0].shape[:2]
    # Banner: top label + metrics text (2 lines).
    banner_h = 66
    label_h = 32
    total_w = W * 2 + 20
    total_h = H + banner_h + label_h

    font_banner = _font(18)
    font_label = _font(22)

    frames = []
    for i in range(n):
        canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        # Top banner with the case name + metric deltas.
        draw.text((10, 6), top_banner_text[0], fill=(255, 230, 120), font=font_banner)
        draw.text((10, 32), top_banner_text[1], fill=(180, 220, 255), font=font_banner)
        # Side labels
        draw.text((W // 2 - 60, banner_h + 4), left_label, fill=(180, 255, 180), font=font_label)
        draw.text((W + 20 + W // 2 - 120, banner_h + 4), right_label, fill=(255, 180, 255), font=font_label)
        # Paste frames
        canvas.paste(Image.fromarray(a[i][:, :, :3]), (0, banner_h + label_h))
        canvas.paste(Image.fromarray(b[i][:, :, :3]), (W + 20, banner_h + label_h))
        frames.append(np.array(canvas))
    imageio.mimsave(out_path, frames, fps=fps, loop=0)


def make_optim_compare(base_json, treat_json, out_path, case):
    with open(base_json) as f:
        rec_b = json.load(f)
    with open(treat_json) as f:
        rec_t = json.load(f)

    metrics = ["psnr", "ssim", "track_err", "yms", "prs"]
    titles = ["PSNR", "SSIM", "Track error", "E (Young's modulus)", "ν (Poisson)"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), dpi=110)

    # Best checkpoint = max PSNR (matches run_pipeline's save logic, which also
    # requires SSIM to increase; PSNR peak is a sufficient visual anchor).
    def best_idx(rec):
        tl = rec.get("test_list", [])
        if not tl:
            return None
        return max(range(len(tl)), key=lambda i: tl[i].get("psnr", 0))

    best_b = best_idx(rec_b)
    best_t = best_idx(rec_t)

    for ax, m, t in zip(axes, metrics, titles):
        for rec, color, label, bi in [(rec_b, "tab:blue", "baseline", best_b),
                                       (rec_t, "tab:orange", "Material + GripMLP", best_t)]:
            tl = rec.get("test_list", [])
            xs = list(range(len(tl)))
            ys = [d.get(m) for d in tl]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5,
                    color=color, label=label)
            if bi is not None:
                # Vertical dashed line + bigger star marker at the best-iter for this run.
                ax.axvline(bi, color=color, linestyle="--", linewidth=0.8, alpha=0.5)
                if ys[bi] is not None:
                    ax.plot(bi, ys[bi], marker="*", markersize=14,
                            markeredgecolor="black", markerfacecolor=color)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("test checkpoint")
        if m in ("yms",):
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    # Master legend at top of figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(case, y=1.08, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_combined_gif(overlay_gif_path, tracking_gif_path, opt_png_path, out_path, fps=8):
    """Stack: optimization PNG on top (static), overlay | tracking gifs on bottom."""
    overlay = imageio.mimread(overlay_gif_path, memtest=False)
    tracking = imageio.mimread(tracking_gif_path, memtest=False)
    opt_img = imageio.imread(opt_png_path)

    n = min(len(overlay), len(tracking))
    # Align bottom row heights by padding the shorter one.
    hA, wA = overlay[0].shape[:2]
    hB, wB = tracking[0].shape[:2]
    bottom_h = max(hA, hB)

    def pad(frame, target_h):
        if frame.shape[0] == target_h:
            return frame
        p = np.full((target_h, frame.shape[1], 3), 20, dtype=np.uint8)
        p[:frame.shape[0]] = frame[..., :3]
        return p

    bottom_w = wA + wB + 20  # 20 px gap
    # Resize optimization PNG to same width, preserving aspect.
    opt_pil = Image.fromarray(opt_img[..., :3])
    aspect = opt_pil.height / opt_pil.width
    opt_h = int(bottom_w * aspect)
    opt_pil = opt_pil.resize((bottom_w, opt_h), Image.LANCZOS)
    opt_arr = np.array(opt_pil)

    total_h = opt_h + bottom_h
    frames = []
    for i in range(n):
        canvas = np.full((total_h, bottom_w, 3), 20, dtype=np.uint8)
        canvas[:opt_h] = opt_arr
        canvas[opt_h:opt_h + hA, :wA] = pad(overlay[i][..., :3], bottom_h)[:hA]
        canvas[opt_h:opt_h + hB, wA + 20:wA + 20 + wB] = pad(tracking[i][..., :3], bottom_h)[:hB]
        frames.append(canvas)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str,
                   default="/home/steven/project-steven/Vid2Sim/comparison_viz")
    p.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for case in args.cases:
        b_dir = os.path.join(BASE, case)
        t_dir = os.path.join(TREAT, case)
        out_case = os.path.join(args.out_dir, case)
        os.makedirs(out_case, exist_ok=True)

        b_best = _best_metrics(os.path.join(b_dir, "optimization_record.json"))
        t_best = _best_metrics(os.path.join(t_dir, "optimization_record.json"))
        psnr_line = (f"PSNR {b_best['psnr']:.2f} → {t_best['psnr']:.2f}  "
                     f"({100*(t_best['psnr']-b_best['psnr'])/b_best['psnr']:+.1f}%)")
        track_line = (f"Track {b_best['track_err']:.3f} → {t_best['track_err']:.3f}  "
                      f"({100*(t_best['track_err']-b_best['track_err'])/b_best['track_err']:+.1f}%)")
        banner = [case, psnr_line + "    " + track_line]

        for gif_name, out_name, crop_left in [
            ("overlay.gif", "overlay_compare.gif", False),
            ("tracking_overlay_v2.gif", "tracking_v2_compare.gif", True),
        ]:
            b_gif = os.path.join(b_dir, gif_name)
            t_gif = os.path.join(t_dir, gif_name)
            if not (os.path.exists(b_gif) and os.path.exists(t_gif)):
                print(f"  skip {case}/{gif_name}")
                continue
            out_path = os.path.join(out_case, out_name)
            make_side_by_side_gif(b_gif, t_gif, out_path,
                                  left_label="Baseline",
                                  right_label="Material + GripMLP",
                                  top_banner_text=banner,
                                  crop_left_half=crop_left)
            print(f"  {out_path}")

        make_optim_compare(os.path.join(b_dir, "optimization_record.json"),
                           os.path.join(t_dir, "optimization_record.json"),
                           os.path.join(out_case, "optimization_compare.png"),
                           case)
        print(f"  {out_case}/optimization_compare.png")

        # Combined single-frame view: optimization plot + both gifs side-by-side.
        overlay_gif = os.path.join(out_case, "overlay_compare.gif")
        tracking_gif = os.path.join(out_case, "tracking_v2_compare.gif")
        opt_png = os.path.join(out_case, "optimization_compare.png")
        if all(os.path.exists(p) for p in (overlay_gif, tracking_gif, opt_png)):
            combined = os.path.join(out_case, "summary.gif")
            make_combined_gif(overlay_gif, tracking_gif, opt_png, combined)
            print(f"  {combined}")
        else:
            print(f"  skip {case}/summary.gif (missing input)")


if __name__ == "__main__":
    main()
