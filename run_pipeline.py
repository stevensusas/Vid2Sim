import sys
import os
import json
import argparse
import random
import imageio
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import tyro
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
from tqdm import trange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import VideoMAEModel
from safetensors.torch import load_file
from PIL import Image

from utils.seeding import seed_everything
from models.phys_predictor import SimulationDataset, FeedForwardPredictor, RegressionHead, LBSHead
from models.lbs_networks import SimplicitsMLP

sys.path.append('LGM')
from LGM.core.models import LGM
from LGM.core.options import AllConfigs  
from kiui.op import recenter

sys.path.append('gs')
from gs.train import training
from gs.arguments import ModelParams, OptimizationParams, PipelineParams
from gs.utils.general_utils import safe_state 

from simulators.lbs_simulator import LBSSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _merge_per_case_sim_config(sim_args, dataset_dir, data_name):
    """Override sim_args with per-case sim_config.yaml if present (e.g. floor convention from PhysTwin)."""
    sim_config_path = os.path.join(dataset_dir, data_name, 'sim_config.yaml')
    if os.path.isfile(sim_config_path):
        case_cfg = OmegaConf.load(sim_config_path)
        sim_args = OmegaConf.merge(sim_args, case_cfg)
        print(f"[Config] Merged per-case sim_config from {sim_config_path}")
    return sim_args


def _project_to_2d(pts_3d, camera):
    """Project 3D points (N, 3) to 2D pixel coords (N, 2) via full_proj_transform."""
    ones = torch.ones(pts_3d.shape[0], 1, device=pts_3d.device)
    pts_h = torch.cat([pts_3d, ones], dim=1)             # (N, 4)
    pts_clip = pts_h @ camera.full_proj_transform         # (N, 4)
    pts_ndc = pts_clip[:, :3] / pts_clip[:, 3:4]          # (N, 3)
    px = (pts_ndc[:, 0] + 1) / 2 * camera.image_width
    py = (pts_ndc[:, 1] + 1) / 2 * camera.image_height
    return torch.stack([px, py], dim=1)                   # (N, 2)


def _run_cotracker_multiview(simulator, dataset_dir, data_name, n_frames=16):
    """Run CoTracker on all views using projected cubature points as queries.
    Returns tracks (V, T, N_cub, 2) in pixel space per view, or None on failure."""
    gs_views = simulator.gs_context['gs_views']
    V = len(gs_views)
    N_cub = simulator.cubature_points.shape[0]

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

    all_tracks = []
    for view_idx in range(V):
        camera = gs_views[view_idx]
        frames = []
        for i in range(n_frames):
            img_path = os.path.join(dataset_dir, data_name, 'data', f'm_{view_idx}_{i}.png')
            if not os.path.exists(img_path):
                print(f"[Tracking] Missing {img_path}, skipping CoTracker.")
                return None
            frames.append(torch.tensor(np.array(Image.open(img_path).convert('RGB'))).permute(2, 0, 1).float())
        video = torch.stack(frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)

        with torch.no_grad():
            proj_2d = _project_to_2d(simulator.cubature_points, camera)  # (N_cub, 2)
            queries = torch.zeros(1, N_cub, 3, device=device)
            queries[0, :, 0] = 0
            queries[0, :, 1:] = proj_2d
            pred_tracks, _ = cotracker(video, queries=queries)  # (1, T, N_cub, 2)
        all_tracks.append(pred_tracks[0].cpu().numpy())  # (T, N_cub, 2)

    tracks = np.stack(all_tracks, axis=0)  # (V, T, N_cub, 2)
    print(f"[Tracking] CoTracker done: {V} views, tracks shape={tracks.shape}")
    del cotracker
    torch.cuda.empty_cache()
    return tracks


def _triangulate_tracks(tracks, cameras):
    """Triangulate multiview 2D tracks to 3D using DLT (vectorized SVD).
    Args:
        tracks:  (V, T, N_cub, 2) pixel coords per view
        cameras: list of V Camera objects
    Returns:
        (T, N_cub, 3) triangulated 3D positions in norm_scale world space
    """
    V, T, N_cub, _ = tracks.shape

    # Projection matrices in standard column-vector convention: P = full_proj_transform.T
    # full_proj_transform is already built from the axis-flipped c2w (readCamerasFromTransforms
    # applies c2w[:3, 1:3] *= -1), so using it directly is correct.
    Ps = [cam.full_proj_transform.cpu().numpy().T for cam in cameras]
    Ws = [cam.image_width  for cam in cameras]
    Hs = [cam.image_height for cam in cameras]

    # Build A: (T, N_cub, 2V, 4)
    A = np.zeros((T, N_cub, 2 * V, 4), dtype=np.float64)
    for v, (P, W, H) in enumerate(zip(Ps, Ws, Hs)):
        u  = tracks[v, :, :, 0]           # (T, N_cub)
        pv = tracks[v, :, :, 1]           # (T, N_cub)
        x_ndc = 2 * u  / W - 1            # (T, N_cub)
        y_ndc = 2 * pv / H - 1            # (T, N_cub)
        A[:, :, 2*v,   :] = x_ndc[:, :, None] * P[3] - P[0]
        A[:, :, 2*v+1, :] = y_ndc[:, :, None] * P[3] - P[1]

    # Batched SVD: (T*N_cub, 2V, 4) -> last right singular vector per point
    A_flat = A.reshape(T * N_cub, 2 * V, 4)
    _, _, Vt = np.linalg.svd(A_flat, full_matrices=False)  # Vt: (T*N_cub, 4, 4)
    X = Vt[:, -1, :]                      # (T*N_cub, 4)
    pts_3d = (X[:, :3] / X[:, 3:4]).reshape(T, N_cub, 3)
    return pts_3d.astype(np.float32)


def _compute_tracking_metric_3d(save_list_pts, triangulated_3d):
    """Mean L2 distance in norm_scale units between simulated and triangulated cubature positions."""
    n_frames = min(len(save_list_pts), triangulated_3d.shape[0])
    frame_errs = [
        np.linalg.norm(save_list_pts[t] - triangulated_3d[t], axis=-1).mean()
        for t in range(n_frames)
    ]
    return float(np.mean(frame_errs))

def predict_phys_params(args):

    # res = 224
    res = 448
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    data_name = args.data_name
    os.makedirs(f'{output_dir}/{data_name}', exist_ok=True)
    
    pred_dataset = SimulationDataset(dataset_dir, data_name, res=res, frame_num=16)
    pred_dataloader = DataLoader(pred_dataset, batch_size=1, num_workers=16, shuffle=False) 
    backbone_model_pretrained = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)

    # Interpolate position embeddings
    if res != 224:
        backbone_model_pretrained.embeddings.patch_embeddings.image_size = (res, res)
        pos_tokens = backbone_model_pretrained.embeddings.position_embeddings
        T = 8
        P = int((pos_tokens.shape[1] // T) ** 0.5)
        C = pos_tokens.shape[2]
        new_P = res // 16
        # B, L, C -> BT, H, W, C -> BT, C, H, W
        pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
        pos_tokens = pos_tokens.reshape(-1, P, P, C).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_P, new_P), mode='bicubic', align_corners=False)
        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
        pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
        backbone_model_pretrained.embeddings.position_embeddings = pos_tokens  # update

    predictor = FeedForwardPredictor(backbone_model_pretrained).to(device)
    predictor.load_state_dict(torch.load(args.ckpt_predictor).state_dict())
    predictor.eval() 
    
    # It only predicts parameters for one object here (you can change it to batch prediction)
    for data in pred_dataloader:
    
        video = data['video'].to(device) 
        output = predictor(video)
        yms_pred, prs_pred, lbs_pred = output['yms'], output['prs'], output['lbs']
  
        print(f"[[Stage I]] Predict physical parameters for {data_name}:")
        print(f"[[Stage I]] Young's Modulus={torch.pow(10, yms_pred).item()} \t Poisson's Ratio={prs_pred.item()}")
        
        params_path = f'{output_dir}/{data_name}/init_params.yaml'
        params = OmegaConf.create()
        params.init_yms = torch.pow(10, yms_pred).item()
        params.init_prs = prs_pred.item() 
        OmegaConf.save(params, params_path)
      
        # Assign predicted LBS weights and biases
        mlp_predict = SimplicitsMLP(spatial_dimensions=3, layer_width=64, num_handles=10, num_layers=8)
        mlp_predict.load_state_dict(torch.load(args.ckpt_lbs))
        mlp_predict.net[-1].weight.data = lbs_pred[0:1, :640].reshape(mlp_predict.net[-1].weight.data.shape)
        mlp_predict.net[-1].bias.data = lbs_pred[0:1, 640:].reshape(mlp_predict.net[-1].bias.data.shape)
        os.makedirs(f'{output_dir}/{data_name}/models', exist_ok=True)
        torch.save(mlp_predict.state_dict(), f'{output_dir}/{data_name}/models/model_pred.pth')

# Modified from LGM/infer.py
def predict_gs_LGM(args): 

    print(f"[[Stage I]] Predict GS for {args.data_name}...") 
     
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    
    sys.argv = ['example.py', 'big']
    opt = tyro.cli(AllConfigs)

    model = LGM(opt)
    ckpt = load_file(args.ckpt_lgm, device='cpu')
    model.load_state_dict(ckpt, strict=False)
    model = model.half().to(device)
    model.eval()
    # bg_remover = rembg.new_session()
    
    rays_embeddings = model.prepare_default_rays(device)
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    images = []
    for view_idx in range(4): 
        
        # We directly use the segmented image here (if your own dataset is not segmented,
        # you can use the rembg (the commented line) or Segment-Anything to remove the background)

        input_image = Image.open(f'{args.dataset_dir}/{args.data_name}/data/a_{view_idx}_0.png')
        input_image = np.array(input_image) 
        # input_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
        mask = input_image[..., -1] > 0

        # Center the masked object in the image
        H, W = input_image.shape[:2]
        coords = np.nonzero(mask)
        if len(coords[0]) > 0:  # Check if mask is not empty
            x_min, x_max, y_min, y_max = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max() 
            mask_center_x, mask_center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            img_center_x, img_center_y = H // 2, W // 2 

            shift_x, shift_y = img_center_x - mask_center_x, img_center_y - mask_center_y 
            centered_image = np.zeros_like(input_image)
            centered_mask = np.zeros_like(mask) 

            for i in range(H):
                for j in range(W):
                    new_i = i + shift_x
                    new_j = j + shift_y
                    if 0 <= new_i < H and 0 <= new_j < W:
                        centered_image[new_i, new_j] = input_image[i, j]
                        centered_mask[new_i, new_j] = mask[i, j]
            input_image = centered_image
            mask = centered_mask

        image = recenter(input_image, mask, border_ratio=0.2) # original LGM operator
        image = image.astype(np.float32) / 255.0  
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        images.append(image)

    mv_image = np.stack(images, axis=0)
    
    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians 
        os.makedirs(f'{args.output_dir}/{args.data_name}/gs_models', exist_ok=True)
        model.gs.save_ply(gaussians, f'{args.output_dir}/{args.data_name}/gs_models/pred.ply')

# Modified from gs/train.py
def refine_gs(args):
    
    print(f"[[Stage II]] Refine GS for {args.data_name} (using standard 3DGS training) ...")  

    gs_parser = argparse.ArgumentParser()
    lp = ModelParams(gs_parser)
    op = OptimizationParams(gs_parser)
    pp = PipelineParams(gs_parser)
    gs_parser.add_argument('--ip', type=str, default="127.0.0.1")
    gs_parser.add_argument('--port', type=int, default=6009)
    gs_parser.add_argument('--debug_from', type=int, default=-1)
    gs_parser.add_argument('--detect_anomaly', action='store_true', default=False)
    gs_parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    gs_parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    gs_parser.add_argument("--quiet", action="store_true")
    gs_parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    gs_parser.add_argument("--start_checkpoint", type=str, default = None)
    gs_args = gs_parser.parse_args(["-s", f'{args.dataset_dir}/{args.data_name}',
                                    "-m", f'{args.output_dir}/{args.data_name}/gs_models',
                                    "--white_background"])
    gs_args.save_iterations.append(gs_args.iterations)
     
    # Initialize system state (RNG)
    safe_state(gs_args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(gs_args.detect_anomaly)
    training(lp.extract(gs_args), op.extract(gs_args), pp.extract(gs_args), gs_args.test_iterations,
        gs_args.save_iterations, gs_args.checkpoint_iterations, gs_args.start_checkpoint, gs_args.debug_from)
 
def refine_lbs(args):

    print(f"[[Stage II]] Refine Neural LBS & Jacobian for {args.data_name} (using data-free training)") 
    with open(f'{args.output_dir}/{args.data_name}/init_params.yaml', 'r') as f:
        init_params = OmegaConf.load(f)
    sim_args = OmegaConf.load(args.config) 
    sim_args = _merge_per_case_sim_config(sim_args, args.dataset_dir, args.data_name)
    sim_args.tag = 'base' 
    sim_args.yms = init_params.init_yms
    sim_args.prs = init_params.init_prs 
    simulator = LBSSimulator(sim_args, args.dataset_dir, args.output_dir, args.data_name)
    simulator.set_material() 
    simulator.refine_lbs()

    # (Optional) Check the simulation results at this stage
    # simulator.load_lbs()
    # simulator.initialize_simulator()
    # simulator.simulate_fast_forward(target_step=15, view_indices=simulator.total_view_indices, render=True)
    # simulator.save_images(simulator.tag)
    # psnr, ssim = simulator.calculate_metrics(view_indices=simulator.total_view_indices, end_step=16)
    # print(f"After refinement -- PSNR: {psnr.item()}, SSIM: {ssim.item()}")

def joint_optimization(args):

    print(f"[[Stage II]] Joint optimization for {args.data_name}")
    with open(f'{args.output_dir}/{args.data_name}/init_params.yaml', 'r') as f:
        init_params = OmegaConf.load(f)
    sim_args = OmegaConf.load(args.config) 
    sim_args = _merge_per_case_sim_config(sim_args, args.dataset_dir, args.data_name)
    sim_args.tag = 'base' 
    sim_args.yms = init_params.init_yms
    sim_args.prs = init_params.init_prs 
    simulator = LBSSimulator(sim_args, args.dataset_dir, args.output_dir, args.data_name, optimization=True)
    simulator.set_material()
    simulator.load_lbs()

    cotracker_tracks = _run_cotracker_multiview(simulator, args.dataset_dir, args.data_name, n_frames=simulator.n_sim_frames)
    if cotracker_tracks is not None:
        triangulated_3d = _triangulate_tracks(cotracker_tracks, simulator.gs_context['gs_views'])
        print(f"[Tracking] Triangulated 3D tracks: shape={triangulated_3d.shape}")
    else:
        triangulated_3d = None

    record = {}
    record['train_list'] = []
    record['test_list'] = []
    best_params = OmegaConf.create()

    # You can try using less optimization iterations and it's sufficient to get a good result
    pbar = trange(sim_args.optimization_iters + 1)
    best_psnr = 0
    best_ssim = 0

    for iter in pbar:

        simulator.initialize_simulator() # Need to re-initialize every iteration since the lbs parameters are updated
        start_step = random.randint(sim_args.simulation_start_step, 11)
        end_step = start_step + 4

        if iter % sim_args.optimization_checkpoint_interval == 0: # Validate the model for dynamic reconstruction
            n_total = simulator.n_sim_frames
            simulator.simulate_fast_forward(n_total - 1, simulator.total_view_indices, render=True)
            psnr, ssim = simulator.calculate_metrics(simulator.total_view_indices, end_step=n_total)
            # Per-point field: summarize by mean over cubatures for logging.
            with torch.no_grad():
                E_field, nu_field = simulator.material_mlp(simulator.cubature_points)
            yms_mean = E_field.mean().item()
            prs_mean = nu_field.mean().item()
            track_err = None
            if triangulated_3d is not None:
                track_err = _compute_tracking_metric_3d(simulator.save_list_pts, triangulated_3d)
            track_str = f" TrackErr={track_err:.4f}" if track_err is not None else ""
            print(f"[Iter {iter}]: PSNR={psnr:.4f} SSIM={ssim:.4f}{track_str}  "
                  f"E_mean={yms_mean:.2f} ν_mean={prs_mean:.4f}")
            entry = {'psnr': psnr.item(), 'ssim': ssim.item(), 'yms': yms_mean, 'prs': prs_mean}
            if track_err is not None:
                entry['track_err'] = track_err
            record['test_list'].append(entry)
            if psnr > best_psnr and ssim > best_ssim: # Save the best model
                best_psnr, best_ssim = psnr, ssim
                record['best_psnr'], record['best_ssim'] = psnr.item(), ssim.item()
                best_params.yms = yms_mean
                best_params.prs = min(prs_mean, 0.49)
                OmegaConf.save(best_params, f'{args.output_dir}/{args.data_name}/best_params.yaml')
                torch.save(simulator.lbs_model.state_dict(), f'{args.output_dir}/{args.data_name}/models/model_best.pth')
                torch.save(simulator.jacobian_model.state_dict(), f'{args.output_dir}/{args.data_name}/models/jmodel_best.pth')
                torch.save(simulator.material_mlp.state_dict(), f'{args.output_dir}/{args.data_name}/models/material_mlp_best.pth')
                if simulator.grip_mlp is not None:
                    torch.save(simulator.grip_mlp.state_dict(), f'{args.output_dir}/{args.data_name}/models/grip_mlp_best.pth')
            simulator.reset_simulator()

        if iter != sim_args.optimization_iters:
            view_indices = torch.arange(0, len(simulator.gs_context['gs_views']), device=device, dtype=torch.long)
            simulator.simulate_fast_forward(start_step, view_indices)
            simulator.simulate_with_grad(end_step, view_indices)
            rendering_loss = simulator.update_parameters(view_indices, start_step, end_step)
            with torch.no_grad():
                E_field, nu_field = simulator.material_mlp(simulator.cubature_points)
            yms_mean = E_field.mean().item()
            prs_mean = nu_field.mean().item()
            record['train_list'].append({'loss': rendering_loss.item(), 'yms': yms_mean, 'prs': prs_mean})

        pbar.set_description(f"Loss={rendering_loss.item()} E_mean={yms_mean:.1f} ν_mean={prs_mean:.3f} ")
        pbar.set_postfix(lr=float(simulator.mat_optimizer.param_groups[0]['lr']))
        with open(f'{args.output_dir}/{args.data_name}/optimization_record.json', 'w') as f:
            json.dump(record, f)

def plot_optimization_record(args):
    record_path = f'{args.output_dir}/{args.data_name}/optimization_record.json'
    with open(record_path, 'r') as f:
        record = json.load(f)

    test_list = record.get('test_list', [])
    train_list = record.get('train_list', [])
    if not test_list:
        print("No optimization record to plot.")
        return

    checkpoint_interval = len(train_list) // max(len(test_list) - 1, 1) if len(test_list) > 1 else 10
    test_iters = [i * checkpoint_interval for i in range(len(test_list))]

    psnr_vals = [d['psnr'] for d in test_list]
    ssim_vals = [d['ssim'] for d in test_list]
    yms_vals  = [d['yms']  for d in test_list]
    prs_vals  = [d['prs']  for d in test_list]
    track_vals = [d['track_err'] for d in test_list if 'track_err' in d]
    has_tracking = len(track_vals) == len(test_list)

    best_psnr = record.get('best_psnr', max(psnr_vals))
    best_ssim = record.get('best_ssim', max(ssim_vals))
    best_idx  = next((i for i, d in enumerate(test_list)
                      if d['psnr'] == best_psnr and d['ssim'] == best_ssim), None)

    if has_tracking:
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        plot_series = [psnr_vals, ssim_vals, track_vals, yms_vals, prs_vals]
        plot_labels = ['PSNR (dB)', 'SSIM', 'Tracking Error (L2)', "Young's Modulus E (Pa)", 'Poisson Ratio ν']
        plot_colors = ['steelblue', 'darkorange', 'firebrick', 'forestgreen', 'mediumpurple']
        ax_list = list(axes.flat)
        ax_list[-1].set_visible(False)  # hide unused 6th cell
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plot_series = [psnr_vals, ssim_vals, yms_vals, prs_vals]
        plot_labels = ['PSNR (dB)', 'SSIM', "Young's Modulus E (Pa)", 'Poisson Ratio ν']
        plot_colors = ['steelblue', 'darkorange', 'forestgreen', 'mediumpurple']
        ax_list = list(axes.flat)

    fig.suptitle(f'Optimization Record — {args.data_name}', fontsize=14)

    for ax, vals, label, color in zip(ax_list, plot_series, plot_labels, plot_colors):
        ax.plot(test_iters, vals, color=color, linewidth=2)
        if best_idx is not None:
            ax.axvline(test_iters[best_idx], color='red', linestyle='--', linewidth=1.2, label=f'Best (iter {test_iters[best_idx]})')
            ax.scatter([test_iters[best_idx]], [vals[best_idx]], color='red', zorder=5)
            ax.legend(fontsize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f'{args.output_dir}/{args.data_name}/optimization_record.png'
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved optimization plot → {out_path}")


def save_overlay_gif(args):
    out_dir   = f'{args.output_dir}/{args.data_name}'
    gt_dir    = f'{args.dataset_dir}/{args.data_name}/data'
    render_dir = f'{out_dir}/render_best'
    n_frames  = len([f for f in os.listdir(render_dir) if f.startswith('r_0_') and f.endswith('.png')])

    gt_frames, rc_frames = [], []
    for i in range(n_frames):
        gt_path = f'{gt_dir}/r_0_{i}.png'
        rc_path = f'{render_dir}/r_0_{i}.png'
        if not os.path.exists(gt_path) or not os.path.exists(rc_path):
            print(f"Missing frame {i}, skipping overlay GIF.")
            return
        gt_frames.append(np.array(Image.open(gt_path).convert('RGB')))
        rc_frames.append(np.array(Image.open(rc_path).convert('RGB')))

    # GT tinted green, recon tinted magenta — misaligned regions show as color fringing
    GT_TINT  = np.array([0.6, 1.0, 0.6], dtype=float)   # greenish
    RC_TINT  = np.array([1.0, 0.6, 1.0], dtype=float)   # magenta

    blended = []
    for gt, rc in zip(gt_frames, rc_frames):
        blend = np.clip(0.5 * gt.astype(float) * GT_TINT + 0.5 * rc.astype(float) * RC_TINT, 0, 255).astype(np.uint8)
        blended.append(blend)

    imageio.mimsave(f'{out_dir}/overlay.gif', blended, fps=8, loop=0)
    print(f"Saved overlay GIF → {out_dir}/overlay.gif")


def save_material_field_gif(simulator, out_path, elev=25, azim=-45):
    """Static 3D scatter of cubature points colored by (log E, ν) from MaterialMLP.

    Two-panel image: left is Young's modulus (log color scale), right is Poisson
    ratio. Uses a single front-right view with some elevation — spinning
    obscured the per-point variation. Color range is auto-scaled to the data
    percentile range so the spatial field stands out, with the MLP's configured
    bounds shown alongside for absolute reference.

    Writes PNG if `out_path` ends in .png, else GIF (one-frame) for compatibility.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import imageio

    # Sample the MaterialMLP at all Gaussian positions (not just cubatures) so
    # the full spatial extent of the field is visible. Cubatures are where the
    # simulator evaluates material forces, but the MLP is a continuous function
    # and interpolates smoothly elsewhere.
    sample_pts = simulator.points
    simulator.material_mlp.eval()
    with torch.no_grad():
        E, nu = simulator.material_mlp(sample_pts)
    pts = sample_pts.detach().cpu().numpy()
    E_np = E.detach().cpu().numpy()
    nu_np = nu.detach().cpu().numpy()
    log_E = np.log10(np.clip(E_np, 1e-6, None))
    bounds = simulator.material_mlp.bounds

    # Auto-scale color to [5, 95] percentile for contrast; clip to MLP bounds.
    def _auto_range(vals, floor_min, floor_max, min_span):
        lo, hi = np.percentile(vals, [5, 95])
        if hi - lo < min_span:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - min_span / 2, mid + min_span / 2
        return max(lo, floor_min), min(hi, floor_max)

    log_E_lo, log_E_hi = _auto_range(log_E, np.log10(bounds["E"][0]),
                                     np.log10(bounds["E"][1]), 0.3)
    nu_lo, nu_hi = _auto_range(nu_np, bounds["nu"][0], bounds["nu"][1], 0.02)

    pad = 0.08 * (pts.max(axis=0) - pts.min(axis=0)).max()
    xmin, ymin, zmin = pts.min(axis=0) - pad
    xmax, ymax, zmax = pts.max(axis=0) + pad

    fig = plt.figure(figsize=(11, 5), dpi=120)
    for panel_idx, (title, vals, vlo, vhi, cmap_name) in enumerate([
        (f"log10(E)  vis=[{log_E_lo:.2f},{log_E_hi:.2f}]  bounds=[{np.log10(bounds['E'][0]):.1f},{np.log10(bounds['E'][1]):.1f}]",
         log_E, log_E_lo, log_E_hi, "viridis"),
        (f"ν  vis=[{nu_lo:.3f},{nu_hi:.3f}]  bounds=[{bounds['nu'][0]:.2f},{bounds['nu'][1]:.2f}]",
         nu_np, nu_lo, nu_hi, "plasma"),
    ]):
        ax = fig.add_subplot(1, 2, panel_idx + 1, projection="3d")
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=vals, cmap=cm.get_cmap(cmap_name), vmin=vlo, vmax=vhi,
                        s=2, alpha=0.6)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
        ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=8)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.05)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)

    if out_path.lower().endswith(".png"):
        imageio.imwrite(out_path, frame)
    else:
        imageio.mimsave(out_path, [frame], fps=1, loop=0)
    print(f"Saved material field viz → {out_path}")


def _save_tracking_3d_gif(save_list_pts, triangulated_3d, floor_level, out_path,
                          dataset_dir=None, data_name=None, n_frames=None, track_err=None,
                          hand_trajectory=None):
    """Render RGB video + GT/simulated cubature points as a 2-panel 3D scatter animation."""
    if len(save_list_pts) == 0:
        return
    T = len(save_list_pts) if n_frames is None else min(n_frames, len(save_list_pts))
    T_gt = triangulated_3d.shape[0] if triangulated_3d is not None else 0
    T_hand = hand_trajectory.shape[0] if hand_trajectory is not None else 0

    all_pts_list = [np.concatenate(save_list_pts[:T], axis=0)]
    if T_gt > 0:
        all_pts_list.append(triangulated_3d[:min(T, T_gt)].reshape(-1, 3))
    if T_hand > 0:
        hand_np = hand_trajectory.detach().cpu().numpy() if hasattr(hand_trajectory, 'detach') else np.asarray(hand_trajectory)
        all_pts_list.append(hand_np[:min(T, T_hand)].reshape(-1, 3))
    all_pts = np.concatenate(all_pts_list, axis=0)
    pad = 0.15
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    zlim = (min(all_pts[:, 2].min() - pad, floor_level - pad), all_pts[:, 2].max() + pad)

    frames = []

    for t in range(T):
        sim_pts = save_list_pts[t]

        fig = plt.figure(figsize=(5, 5), dpi=100)
        fig.patch.set_facecolor('white')
        ax3d = fig.add_subplot(1, 1, 1, projection='3d')
        ax3d.set_facecolor('#f5f5f5')
        if t < T_gt:
            gt_pts = triangulated_3d[t]
            ax3d.scatter(gt_pts[:, 0],  gt_pts[:, 1],  gt_pts[:, 2],
                         c='green', s=4, alpha=0.7, label='GT (CoTracker)')
        ax3d.scatter(sim_pts[:, 0], sim_pts[:, 1], sim_pts[:, 2],
                     c='red',   s=4, alpha=0.7, label='Simulated')
        if t < T_hand:
            hp = hand_np[t]
            ax3d.scatter(hp[:, 0], hp[:, 1], hp[:, 2],
                         c='blue', s=30, alpha=0.9, marker='X', label='Hand')

        xs = np.linspace(xlim[0], xlim[1], 4)
        ys = np.linspace(ylim[0], ylim[1], 4)
        xx, yy = np.meshgrid(xs, ys)
        ax3d.plot_surface(xx, yy, np.full_like(xx, floor_level),
                          alpha=0.2, color='saddlebrown')
        ax3d.set_xlim(xlim); ax3d.set_ylim(ylim); ax3d.set_zlim(zlim)
        ax3d.set_xlabel('X', fontsize=7); ax3d.set_ylabel('Y', fontsize=7); ax3d.set_zlabel('Z', fontsize=7)
        ax3d.tick_params(labelsize=6)
        ax3d.view_init(elev=20, azim=45)
        title_3d = 'Cubature Points (3D)'
        if track_err is not None:
            title_3d += f'  |  TrackErr={track_err:.4f}'
        ax3d.set_title(title_3d, fontsize=9)
        ax3d.legend(loc='upper right', fontsize=7, markerscale=2)

        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(buf.reshape(h, w, 4)[:, :, :3])
        plt.close(fig)

    if frames:
        imageio.mimsave(out_path, frames, fps=8, loop=0)
        print(f"Saved tracking 3D GIF → {out_path}")


def _save_tracking_gif_impl(args, save_list_pts, cotracker_tracks, camera, out_path,
                            n_frames=None, dot_radius=2, track_err=None, view_idx=0,
                            hand_trajectory=None):
    """Implementation: draw tracking dots on frames and save GIF."""
    if len(save_list_pts) == 0:
        return
    from PIL import ImageDraw, ImageFont

    frames_out = []
    T = len(save_list_pts) if n_frames is None else min(n_frames, len(save_list_pts))
    T_gt = cotracker_tracks.shape[1] if cotracker_tracks is not None else 0
    T_hand = hand_trajectory.shape[0] if hand_trajectory is not None else 0
    gt_dir = os.path.join(args.dataset_dir, args.data_name, 'data')
    sim_dir = os.path.join(args.output_dir, args.data_name, 'render_best')

    W, H = camera.image_width, camera.image_height
    legend_h = 36
    pad = 6

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except Exception:
        font = ImageFont.load_default()

    # Pre-compute projected sim and hand points per frame
    sim_proj_per_frame = []
    hand_proj_per_frame = []
    for t in range(T):
        sim_pts = torch.tensor(save_list_pts[t], device=device)
        sim_proj_per_frame.append(_project_to_2d(sim_pts, camera).cpu().numpy())
        if t < T_hand:
            hand_pts = hand_trajectory[t].to(device) if hasattr(hand_trajectory, 'to') else torch.tensor(hand_trajectory[t], device=device)
            hand_proj_per_frame.append(_project_to_2d(hand_pts, camera).cpu().numpy())
        else:
            hand_proj_per_frame.append(None)

    def _draw_dots(img, gt_pts, sim_proj, hand_proj):
        draw = ImageDraw.Draw(img)
        if gt_pts is not None:
            for (x, y) in gt_pts:
                x, y = float(x), float(y)
                if np.isfinite(x) and np.isfinite(y):
                    draw.ellipse([x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius], fill=(0, 220, 0))
        for (x, y) in sim_proj:
            x, y = float(x), float(y)
            if np.isfinite(x) and np.isfinite(y):
                draw.ellipse([x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius], fill=(220, 0, 0))
        if hand_proj is not None:
            hr = dot_radius + 2
            for (x, y) in hand_proj:
                x, y = float(x), float(y)
                if np.isfinite(x) and np.isfinite(y):
                    draw.ellipse([x-hr, y-hr, x+hr, y+hr], fill=(50, 120, 255))
        return img

    for t in range(T):
        gt_img_path  = os.path.join(gt_dir,  f'm_{view_idx}_{t}.png')
        sim_img_path = os.path.join(sim_dir, f'r_{view_idx}_{t}.png')
        if not os.path.exists(gt_img_path):
            continue

        gt_img  = Image.open(gt_img_path).convert('RGB').copy()
        sim_img = Image.open(sim_img_path).convert('RGB').copy() if os.path.exists(sim_img_path) else Image.new('RGB', (W, H), (200, 200, 200))

        gt_pts = cotracker_tracks[view_idx, t] if t < T_gt else None
        sim_proj = sim_proj_per_frame[t]
        hand_proj = hand_proj_per_frame[t]

        gt_img  = _draw_dots(gt_img,  gt_pts, sim_proj, hand_proj)
        sim_img = _draw_dots(sim_img, gt_pts, sim_proj, hand_proj)

        # Stitch side by side with legend bar
        canvas = Image.new('RGB', (W * 2, H + legend_h), (30, 30, 30))
        canvas.paste(gt_img,  (0, 0))
        canvas.paste(sim_img, (W, 0))
        ldraw = ImageDraw.Draw(canvas)

        # Column labels
        ldraw.text((pad, H + pad), 'GT Video', fill=(200, 200, 200), font=font)
        ldraw.text((W + pad, H + pad), 'Simulated', fill=(200, 200, 200), font=font)

        # Legend swatches
        legend_x = W // 2 - 160
        ldraw.rectangle([legend_x, H + pad, legend_x + 14, H + pad + 14], fill=(0, 220, 0))
        ldraw.text((legend_x + 18, H + pad), 'GT (CoTracker)', fill=(255, 255, 255), font=font)
        legend_x2 = legend_x + 160
        ldraw.rectangle([legend_x2, H + pad, legend_x2 + 14, H + pad + 14], fill=(220, 0, 0))
        ldraw.text((legend_x2 + 18, H + pad), 'Simulated', fill=(255, 255, 255), font=font)
        if hand_trajectory is not None:
            legend_x3 = legend_x2 + 120
            ldraw.rectangle([legend_x3, H + pad, legend_x3 + 14, H + pad + 14], fill=(50, 120, 255))
            ldraw.text((legend_x3 + 18, H + pad), 'Hand', fill=(255, 255, 255), font=font)

        if track_err is not None:
            ldraw.text((W + pad, H + legend_h // 2 + 2), f'TrackErr={track_err:.4f}', fill=(255, 220, 50), font=font)

        frames_out.append(np.array(canvas))

    if frames_out:
        imageio.mimsave(out_path, frames_out, fps=8, loop=0)
        print(f"Saved tracking GIF → {out_path}")


def final_simulation(args):

    print(f"[[Stage II]] Final simulation for {args.data_name}")
    with open(f'{args.output_dir}/{args.data_name}/best_params.yaml', 'r') as f:
        best_params = OmegaConf.load(f)
    sim_args = OmegaConf.load(args.config) 
    sim_args = _merge_per_case_sim_config(sim_args, args.dataset_dir, args.data_name)
    sim_args.tag = 'best' 
    sim_args.yms = best_params.yms
    sim_args.prs = best_params.prs 
    simulator = LBSSimulator(sim_args, args.dataset_dir, args.output_dir, args.data_name)
    simulator.set_material()
    # Load trained MaterialMLP / GripMLP state dicts so the eval sim uses the
    # per-point field that was learned during Stage II joint optimization.
    mat_mlp_path = f'{args.output_dir}/{args.data_name}/models/material_mlp_best.pth'
    if os.path.exists(mat_mlp_path):
        simulator.material_mlp.load_state_dict(torch.load(mat_mlp_path, map_location=simulator.device))
        print(f"[Final] Loaded MaterialMLP from {mat_mlp_path}")
    grip_mlp_path = f'{args.output_dir}/{args.data_name}/models/grip_mlp_best.pth'
    if simulator.grip_mlp is not None and os.path.exists(grip_mlp_path):
        simulator.grip_mlp.load_state_dict(torch.load(grip_mlp_path, map_location=simulator.device))
        print(f"[Final] Loaded GripMLP from {grip_mlp_path}")
    simulator.load_lbs()
    simulator.initialize_simulator()

    cotracker_tracks = _run_cotracker_multiview(simulator, args.dataset_dir, args.data_name, n_frames=simulator.n_sim_frames)
    if cotracker_tracks is not None:
        triangulated_3d = _triangulate_tracks(cotracker_tracks, simulator.gs_context['gs_views'])
    else:
        triangulated_3d = None

    # Simulate the full sequence (scales with hand trajectory length for interaction cases)
    n_total = simulator.n_sim_frames
    simulator.simulate_fast_forward(target_step=n_total - 1, view_indices=simulator.total_view_indices, render=True)
    simulator.save_images(simulator.tag)
    psnr, ssim = simulator.calculate_metrics(view_indices=simulator.total_view_indices, end_step=n_total)
    track_err = None
    if triangulated_3d is not None:
        track_err = _compute_tracking_metric_3d(simulator.save_list_pts, triangulated_3d)
    track_str = f", TrackErr={track_err:.4f}" if track_err is not None else ""
    print(f"After joint optimization -- PSNR: {psnr.item():.4f}, SSIM: {ssim.item():.4f}{track_str}")

    # Load best iter from optimization record
    best_iter = '?'
    record_path = f'{args.output_dir}/{args.data_name}/optimization_record.json'
    if os.path.exists(record_path):
        with open(record_path) as f:
            record = json.load(f)
        best_psnr_rec = record.get('best_psnr', None)
        best_ssim_rec = record.get('best_ssim', None)
        checkpoint_interval = len(record.get('train_list', [])) // max(len(record.get('test_list', [1])) - 1, 1)
        for i, d in enumerate(record.get('test_list', [])):
            if d['psnr'] == best_psnr_rec and d['ssim'] == best_ssim_rec:
                best_iter = i * checkpoint_interval
                break

    annotation = dict(iter=best_iter, E=best_params.yms, nu=best_params.prs,
                      psnr=psnr.item(), ssim=ssim.item())
    simulator.save_pointcloud_gif(simulator.tag, annotation=annotation)

    save_material_field_gif(
        simulator,
        f'{args.output_dir}/{args.data_name}/material_field.gif',
    )

    # Visualize the simulation results at front view
    output_dir = f'{args.output_dir}/{args.data_name}/render_best'
    gt_dir = f'{args.dataset_dir}/{args.data_name}/data'
    pred_imgs, gt_imgs = [], []
    for i in range(simulator.n_sim_frames):
        pred_imgs.append(imageio.imread(f'{output_dir}/r_0_{i}.png'))
        gt_imgs.append(imageio.imread(f'{gt_dir}/r_0_{i}.png'))
    pred_imgs = np.stack(pred_imgs, axis=0)
    gt_imgs = np.stack(gt_imgs, axis=0) 
    imageio.mimsave(f'{args.output_dir}/{args.data_name}/recon.gif', pred_imgs, fps=20, loop=0)
    imageio.mimsave(f'{args.output_dir}/{args.data_name}/gt.gif', gt_imgs, fps=20, loop=0)

    plot_optimization_record(args)
    save_overlay_gif(args)
    hand_traj = simulator.hand_trajectory if simulator.hand_trajectory is not None else None
    if cotracker_tracks is not None or hand_traj is not None:
        gs_views = simulator.gs_context['gs_views']
        for view_idx, camera in enumerate(gs_views):
            suffix = '' if view_idx == 0 else f'_v{view_idx}'
            track_gif_path = f'{args.output_dir}/{args.data_name}/tracking_overlay{suffix}.gif'
            _save_tracking_gif_impl(args, simulator.save_list_pts, cotracker_tracks,
                                    camera, track_gif_path, track_err=track_err,
                                    view_idx=view_idx, hand_trajectory=hand_traj)
        track_3d_path = f'{args.output_dir}/{args.data_name}/tracking_3d.gif'
        _save_tracking_3d_gif(simulator.save_list_pts, triangulated_3d,
                              simulator.floor_level, track_3d_path,
                              dataset_dir=args.dataset_dir, data_name=args.data_name,
                              track_err=track_err, hand_trajectory=hand_traj)

def run_recon(args):

    ### [Stage I] Predict initial physical parameters from single view video & reconstruct GS with pretrained LGM 
    predict_phys_params(args)
    predict_gs_LGM(args)

    ### [Stage II] Refine GS, Neural LBS and Neural Jacobian before joint optimization
    refine_gs(args)
    refine_lbs(args)

    ### [Stage II] Joint optimization
    joint_optimization(args)
    final_simulation(args)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gso.yaml") 
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--data_name", type=str, default="bus")
    parser.add_argument("--ckpt_predictor", type=str, default="checkpoints/ckpt_phys_predictor.pth")
    parser.add_argument("--ckpt_lbs", type=str, default="checkpoints/ckpt_lbs_template.pth")
    parser.add_argument("--ckpt_lgm", type=str, default="checkpoints/ckpt_lgm.safetensors")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
 
    seed_everything(args.seed)
    run_recon(args)