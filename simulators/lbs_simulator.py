import kaolin.physics as physics
import numpy as np
import torch
import torchvision
import os  
import gc 

from functools import partial
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from models.lbs_networks import SimplicitsMLP, JacobianMLP
from simulators.sim_utils.loading import load_points_from_mesh, load_gaussians, load_gts, load_cubature_points
from simulators.sim_utils.physics import newton_E, newton_G, newton_H, train_step_lbs, train_step_jacobian 
from simulators.sim_utils.gaussians import render_gs, build_covariance_from_scaling_rotation_deformations
from simulators.sim_utils.optimization import newtons_method 
from simulators.sim_utils.material import NeohookeanMaterial
from tqdm import tqdm
 
class LBSSimulator():

    def __init__(self, args, dataset_dir, output_dir, data_name, optimization=False, produce_data=False):
        
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.tag = args.tag
        self.optimization = optimization
        self.produce_data = produce_data

        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.data_name = data_name

        self.floor_level = args.floor_level - 0.90
        self.floor_axis = args.floor_axis
        self.flip_floor = getattr(args, 'flip_floor', False)
        self.delta_t = args.delta_t
        self.particle_volume = args.particle_volume
        self.rho = args.rho
        self.penalty_weight = args.penalty_weight
        self.num_cubature_points = args.num_cubature_points
        self.num_handles = args.num_handles
        self.refine_lbs_iters = args.refine_lbs_iters
        self.refine_lbs_samples = args.refine_lbs_samples 
        self.refine_lbs_batch_size = args.refine_lbs_batch_size
        self.refine_jacobian_batch_size = args.refine_jacobian_batch_size
        self.refine_jacobian_samples = args.refine_jacobian_samples  
        self.simulation_newton_iters = args.simulation_newton_iters

        if args.model_type == 'gs':
            gaussians, gs_context = load_gaussians(dataset_dir, output_dir, data_name, self.tag)
        elif args.model_type == 'mesh': 
            mesh_path = f'{dataset_dir}/{data_name}/mesh.glb'
            gaussians, gs_context = None, None
        else:
            raise ValueError(f'Unknown model type: {args.model_type}')
        
        if args.model_type == 'gs':
            self.points = gaussians.get_xyz.clone().detach().to(device)
            self.total_view_indices = torch.arange(0, len(gs_context['gs_views']), 1)
            self.ref = load_gts(dataset_dir, data_name, args.view_samples, 24) 
        elif args.model_type == 'mesh':
            self.points, mesh = load_points_from_mesh(mesh_path)
            self.total_view_indices = torch.arange(0, 1, 1) 
            self.ref = None
         
        self.points = self.points.clone()
        self.points[:, self.floor_axis] += 0.0

        self.cubature_points, self.cubature_points_idx = load_cubature_points(self.points, self.num_cubature_points)
        self.points = self.points.to(device)

        if gaussians is not None:
            self.gaussians = gaussians 
            self.gs_context = gs_context  

        self.lbs_model = SimplicitsMLP(spatial_dimensions=3, layer_width=64,
            num_handles=self.num_handles, num_layers=8).to(device)
        self.jacobian_model = JacobianMLP(spatial_dimensions=3, layer_width=1024,
            num_handles=self.num_handles+1, num_layers=3).to(device)
        self.dFdz = None

        self.loss_fn = torch.nn.MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
 
    def set_material(self):
 
        self.point_wise_rho = torch.ones_like(self.points[:, 0:1]) * self.rho
        self.cubature_point_wise_rho = torch.ones_like(self.cubature_points[:, 0:1]) * self.rho 

        if self.optimization: 
            init_yms = self.args.yms
            init_prs = self.args.prs 
            self.pred_yms_normalized = torch.nn.Parameter(torch.tensor(np.log10(init_yms), device=self.device).float())
            self.pred_prs_normalized = torch.nn.Parameter(torch.tensor(init_prs, device=self.device).float()) 
            self.mat_optimizer = torch.optim.Adam([{'params': [self.pred_yms_normalized], 'lr': 5e-3}, 
                                                {'params': [self.pred_prs_normalized], 'lr': 1e-3},
                                                {'params': self.lbs_model.parameters(), 'lr': 5e-7},
                                                {'params': self.jacobian_model.parameters(), 'lr': 5e-7}])
            print(f'[Joint Optimization] Initial Material Parameters: E={init_yms} ν={init_prs}')
        else:
            self.yms, self.prs = self.args.yms, min(self.args.prs, 0.49)
            self.point_wise_yms = torch.ones_like(self.points[:, 0:1]) * self.yms
            self.point_wise_prs = torch.ones_like(self.points[:, 0:1]) * self.prs
            self.cubature_point_wise_yms = torch.ones_like(self.cubature_points[:, 0:1]) * self.yms
            self.cubature_point_wise_prs = torch.ones_like(self.cubature_points[:, 0:1]) * self.prs 
            self.pred_yms_normalized = None
            self.pred_prs_normalized = None
            print(f'[Simulation] Material Parameters: E={self.yms} ν={self.prs}')
             
        self.grav = torch.zeros(3, device=self.device)
        self.grav[self.floor_axis] = 7.5
    
    def set_lbs(self):
        self.lbs_model_plus_rigid = lambda x: torch.cat((self.lbs_model(x),
            torch.ones((x.shape[0], 1), device=self.device)), dim=1)
        self.skinning_weights = self.lbs_model_plus_rigid(self.points)
        self.skinning_weights_cubature = self.lbs_model_plus_rigid(self.cubature_points) 

    def refine_lbs(self, skip_lbs=False, skip_jacobian=False):
        
        os.makedirs(f'{self.output_dir}/{self.data_name}/models', exist_ok=True)

        # Refine LBS Model (data-free training)
        lbs_path = f'{self.output_dir}/{self.data_name}/models/model_{self.tag}.pth' 
        if skip_lbs and os.path.exists(lbs_path):
            self.lbs_model.load_state_dict(torch.load(lbs_path))
        else: 

            if self.produce_data: 
                self.lbs_model_optimizer = torch.optim.Adam(
                    [{'params': list(self.lbs_model.net[-1].parameters()), 'lr': 1e-3}]
                )
            else:
                self.lbs_model_optimizer = torch.optim.Adam([{'params': self.lbs_model.parameters(), 'lr': 1e-3}])         
            
            self.lbs_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.lbs_model_optimizer, T_max=self.refine_lbs_iters, eta_min=0)
            train_step_lbs_fn = partial(train_step_lbs, model=self.lbs_model, normalized_pts=self.points,
                batch_size=self.refine_lbs_batch_size, num_handles=self.num_handles, appx_vol=self.particle_volume, 
                yms=self.point_wise_yms, prs=self.point_wise_prs, num_samples=self.refine_lbs_samples, 
                le_coeff=1e-1, lo_coeff=1e6)
            
            self.lbs_model.train()
            pbar = tqdm(range(self.refine_lbs_iters))
            for i in pbar:
                le, lo = train_step_lbs_fn(en_interp=float(i / self.refine_lbs_iters))
                loss = le + lo
                self.lbs_model_optimizer.zero_grad()
                loss.backward()
                self.lbs_model_optimizer.step()
                self.lbs_model_scheduler.step()
                if i % 50 == 0:
                    pbar.set_description(f'le: {le.item()}, lo: {lo.item()}')
                    pbar.set_postfix({'lr': self.lbs_model_optimizer.param_groups[0]['lr']})
            torch.save(self.lbs_model.state_dict(), lbs_path)
        
        # Train Jacobian Model (data-free training)
        if not self.produce_data:
            jacobian_path = f'{self.output_dir}/{self.data_name}/models/jmodel_{self.tag}.pth'
            if skip_jacobian and os.path.exists(jacobian_path):
                self.jacobian_model.load_state_dict(torch.load(jacobian_path))
            else: 
                self.jacobian_model_optimizer = torch.optim.AdamW(self.jacobian_model.parameters(), lr=1e-3, betas=(0.9, 0.98))
                self.jacobian_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.jacobian_model_optimizer,
                    T_max=self.refine_lbs_iters, eta_min=0)
                train_step_jacobian_fn = partial(train_step_jacobian, model=self.lbs_model, j_model=self.jacobian_model,
                    normalized_pts=self.points, batch_size=self.refine_jacobian_batch_size, num_handles=self.num_handles+1,
                    num_samples=self.refine_jacobian_samples)
                
                self.jacobian_model.train()
                pbar = tqdm(range(self.refine_lbs_iters))
                for i in pbar:
                    loss = train_step_jacobian_fn()
                    self.jacobian_model_optimizer.zero_grad()
                    loss.backward()
                    self.jacobian_model_optimizer.step()
                    self.jacobian_model_scheduler.step()
                    if i % 50 == 0:
                        pbar.set_description(f'loss: {loss.item()}')
                        pbar.set_postfix({'lr': self.jacobian_model_optimizer.param_groups[0]['lr']})
                torch.save(self.jacobian_model.state_dict(), jacobian_path)

        if self.optimization or self.tag == 'best':
            self.use_neural_jacobian = True
        else:
            self.use_neural_jacobian = False
        self.set_lbs()

    def load_lbs(self):
        if os.path.exists(f'{self.output_dir}/{self.data_name}/models/model_{self.tag}.pth'):
            self.lbs_model.load_state_dict(torch.load(f'{self.output_dir}/{self.data_name}/models/model_{self.tag}.pth'))
            if self.optimization or self.tag == 'best':
                self.jacobian_model.load_state_dict(torch.load(f'{self.output_dir}/{self.data_name}/models/jmodel_{self.tag}.pth'))
                self.use_neural_jacobian = True
            else:
                self.use_neural_jacobian = False
            self.set_lbs()
        else:
            self.refine_lbs()

    def reset_simulator(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.z = torch.zeros(self.skinning_weights_cubature.shape[1] * 12,
            device=self.device, dtype=torch.float32).unsqueeze(-1)
        self.z_dot = torch.zeros_like(self.z, device=self.device) 
        self.current_step = 0
        self.save_list = []
        self.save_list_bg = []
        self.save_list_pts = []

    def initialize_simulator(self):
 
        self.x0_flat = self.cubature_points.flatten().unsqueeze(-1)
        self.x0_flat_full = self.points.flatten().unsqueeze(-1)
        self.x0_flat.requires_grad_(True)

        self.skinning_weights = self.lbs_model_plus_rigid(self.points)
        self.skinning_weights_cubature = self.lbs_model_plus_rigid(self.cubature_points)

        if self.optimization:
            self.cubature_point_wise_yms = torch.ones_like(self.cubature_points[:, 0:1]) * torch.pow(10, self.pred_yms_normalized)
            clamped_prs = torch.clamp(self.pred_prs_normalized, -0.99, 0.49)
            self.cubature_point_wise_prs = torch.ones_like(self.cubature_points[:, 0:1]) * clamped_prs
        
        if self.dFdz is None or self.optimization:
            self.M, self.invM = physics.simplicits.precomputed.lumped_mass_matrix(self.cubature_point_wise_rho, self.particle_volume, dim=3)
            self.B = physics.simplicits.precomputed.lbs_matrix(self.cubature_points, self.skinning_weights_cubature)
            self.B_full = physics.simplicits.precomputed.lbs_matrix(self.points, self.skinning_weights)
            self.BMB = self.B.T @ self.M @ self.B
            
            if self.use_neural_jacobian:
                self.dFdz = self.jacobian_model(self.cubature_points).reshape(self.num_cubature_points * 9, -1)
            else: 
                self.dFdz = physics.simplicits.precomputed.jacobian_dF_dz(self.lbs_model_plus_rigid, self.cubature_points, 
                    z=torch.zeros(self.skinning_weights.shape[1]*12, device=self.device).unsqueeze(-1))
            
            self.bigI = torch.tile(torch.eye(3, device=self.device).flatten().unsqueeze(dim=1), (self.num_cubature_points, 1))
 
        self.material_object = NeohookeanMaterial(self.cubature_point_wise_yms, self.cubature_point_wise_prs) 
        self.gravity_object = physics.utils.Gravity(rhos=self.cubature_point_wise_rho, acceleration=self.grav)
        self.floor_object = physics.utils.Floor(floor_height=self.floor_level, floor_axis=self.floor_axis, flip_floor=self.flip_floor)
        self.integration_sampling = torch.as_tensor(self.particle_volume / self.num_cubature_points,
            device=self.device, dtype=torch.float32) 

        self.partial_floor_e = physics.simplicits.simplicits_scene_forces.generate_fcn_simplicits_scene_energy(self.floor_object,
            self.B, coeff=self.penalty_weight, integration_sampling=None)
        self.partial_grav_e = physics.simplicits.simplicits_scene_forces.generate_fcn_simplicits_scene_energy(self.gravity_object,
            self.B, coeff=1, integration_sampling=self.integration_sampling)
        self.partial_material_e = physics.simplicits.simplicits_scene_forces.generate_fcn_simplicits_material_energy(self.material_object,
            self.dFdz, coeff=1, integration_sampling=self.integration_sampling)
    
        self.partial_floor_g = physics.simplicits.simplicits_scene_forces.generate_fcn_simplicits_scene_gradient(self.floor_object,
            self.B, coeff=self.penalty_weight, integration_sampling=None)
        self.partial_grav_g = physics.simplicits.simplicits_scene_forces.generate_fcn_simplicits_scene_gradient(self.gravity_object,
            self.B, coeff=1, integration_sampling=self.integration_sampling)
        self.partial_material_g = physics.simplicits.simplicits_scene_forces.generate_fcn_simplicits_material_gradient(self.material_object,
            self.dFdz, coeff=1, integration_sampling=self.integration_sampling)

        self.partial_floor_h = physics.simplicits.simplicits_scene_forces.generate_fcn_simplicits_scene_hessian(self.floor_object,
            self.B, coeff=self.penalty_weight, integration_sampling=None)
        self.partial_grav_h = physics.simplicits.simplicits_scene_forces.generate_fcn_simplicits_scene_hessian(self.gravity_object,
            self.B, coeff=1, integration_sampling=self.integration_sampling)
        self.partial_material_h = physics.simplicits.simplicits_scene_forces.generate_fcn_simplicits_material_hessian(self.material_object,
            self.dFdz, coeff=1, integration_sampling=self.integration_sampling)
        
        self.partial_newton_E = partial(newton_E, B=self.B, BMB=self.BMB, dt=self.delta_t, x0_flat=self.x0_flat,
            dFdz=self.dFdz, bigI=self.bigI, model=self.lbs_model_plus_rigid, defo_grad_energies=[self.partial_material_e],
            pt_wise_energies=[self.partial_grav_e, self.partial_floor_e])
        self.partial_newton_G = partial(newton_G, B=self.B, BMB=self.BMB, dt=self.delta_t, x0_flat=self.x0_flat,
            dFdz=self.dFdz, bigI=self.bigI, model=self.lbs_model_plus_rigid, defo_grad_gradients=[self.partial_material_g],
            pt_wise_gradients=[self.partial_grav_g, self.partial_floor_g])
        self.partial_newton_H = partial(newton_H, B=self.B, BMB=self.BMB, dt=self.delta_t, x0_flat=self.x0_flat,
            dFdz=self.dFdz, bigI=self.bigI, model=self.lbs_model_plus_rigid, defo_grad_hessians=[self.partial_material_h],
            pt_wise_hessians=[self.partial_grav_h, self.partial_floor_h]) 
            
        self.reset_simulator()

    def render_frame(self, step, views):

        points_flat = self.points.flatten().unsqueeze(-1)
        if step == 0:
            x_pts_full = points_flat.reshape(-1, 3)
        else:
            x_pts_full = (self.B_full @ self.z + points_flat).reshape(-1, 3)

        if self.args.model_type == 'gs': # Transform 3DGS
            self.gaussians._xyz = x_pts_full
            partial_weight_fcn_lbs = partial(physics.simplicits.utils.weight_function_lbs,
                tfms=self.z.reshape(-1,3,4).unsqueeze(0), fcn=self.lbs_model_plus_rigid)
            F_z = physics.utils.finite_diff.finite_diff_jac(partial_weight_fcn_lbs, self.points).squeeze()
            build_cov = partial(build_covariance_from_scaling_rotation_deformations, defo_grad=F_z)
            self.gaussians.covariance_activation = build_cov
            renderings, rendering_bg = render_gs(self.gaussians, views, self.gs_context['gs_pipeline'],
                self.gs_context['gs_background'], self.dataset_dir, self.data_name)
            self.save_list.append(torch.stack(renderings, dim=0))
            self.save_list_bg.append(torch.stack(rendering_bg, dim=0))
            if step == 0:
                x_cubature = self.cubature_points
            else:
                x_cubature = (self.B @ self.z + self.x0_flat).reshape(-1, 3)
            self.save_list_pts.append(x_cubature.clone().detach().cpu().numpy())
        elif self.args.model_type == 'mesh':
            self.save_list.append(x_pts_full.clone().detach().cpu().numpy())

    def save_images(self, save_name): 
        save_path = f'{self.output_dir}/{self.data_name}/render_{save_name}'
        renderings = torch.stack(self.save_list, dim=1)
        renderings_bg = torch.stack(self.save_list_bg, dim=1)
        os.makedirs(save_path, exist_ok=True)
        for view_idx in range(renderings.shape[0]):
            for frame_idx in range(renderings.shape[1]):
                torchvision.utils.save_image(renderings[view_idx, frame_idx], f'{save_path}/m_{view_idx}_{frame_idx}.png')
                torchvision.utils.save_image(renderings_bg[view_idx, frame_idx], f'{save_path}/r_{view_idx}_{frame_idx}.png')

    def save_pointcloud_gif(self, save_name, fps=20, annotation=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import imageio
        from PIL import Image

        save_path = f'{self.output_dir}/{self.data_name}/render_{save_name}'
        out_path  = f'{self.output_dir}/{self.data_name}/recon_pts.gif'
        gt_dir    = f'{self.dataset_dir}/{self.data_name}/data'

        all_pts = np.concatenate(self.save_list_pts, axis=0)
        pad  = 0.15
        xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
        zlim = (min(all_pts[:, 2].min() - pad, self.floor_level - pad), all_pts[:, 2].max() + pad)

        GT_TINT = np.array([0.6, 1.0, 0.6], dtype=float)
        RC_TINT = np.array([1.0, 0.6, 1.0], dtype=float)

        if annotation:
            annot_text = (f"Best Iter: {annotation.get('iter', '?')}    "
                          f"E = {annotation.get('E', 0):.0f} Pa    "
                          f"ν = {annotation.get('nu', 0):.4f}    "
                          f"PSNR = {annotation.get('psnr', 0):.2f} dB    "
                          f"SSIM = {annotation.get('ssim', 0):.4f}")
        else:
            annot_text = ''

        BG = 'white'
        LABEL_COLOR = '#222222'
        LABEL_FS = 13

        frames = []
        for frame_idx, pts in enumerate(self.save_list_pts):

            # ── load images ──────────────────────────────────────────────────
            gt_path = f'{gt_dir}/r_0_{frame_idx}.png'
            rc_path = f'{save_path}/r_0_{frame_idx}.png'
            blank   = np.full((448, 448, 3), 200, dtype=np.uint8)
            gt_img  = np.array(Image.open(gt_path).convert('RGB')) if os.path.exists(gt_path) else blank
            rc_img  = np.array(Image.open(rc_path).convert('RGB')) if os.path.exists(rc_path) else blank
            overlay = np.clip(0.5 * gt_img.astype(float) * GT_TINT +
                              0.5 * rc_img.astype(float) * RC_TINT, 0, 255).astype(np.uint8)
            H, W = gt_img.shape[:2]

            # ── render 3-D point cloud to a numpy image ───────────────────────
            fig3d = plt.figure(figsize=(W / 100, H / 100), dpi=100)
            fig3d.patch.set_facecolor(BG)
            ax3d = fig3d.add_subplot(1, 1, 1, projection='3d')
            ax3d.set_facecolor('#f5f5f5')
            stride = max(1, len(pts) // 2000)
            p = pts[::stride]
            ax3d.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 2], cmap='viridis', s=1, alpha=0.7)
            xs, ys = np.linspace(*xlim, 4), np.linspace(*ylim, 4)
            xx, yy = np.meshgrid(xs, ys)
            ax3d.plot_surface(xx, yy, np.full_like(xx, self.floor_level),
                              alpha=0.25, color='saddlebrown')
            ax3d.set_xlim(xlim); ax3d.set_ylim(ylim); ax3d.set_zlim(zlim)
            ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
            ax3d.tick_params(labelsize=6)
            ax3d.view_init(elev=20, azim=45)
            fig3d.tight_layout(pad=0.2)
            fig3d.canvas.draw()
            buf3d = np.frombuffer(fig3d.canvas.buffer_rgba(), dtype=np.uint8)
            w3d, h3d = fig3d.canvas.get_width_height()
            pc_img = buf3d.reshape(h3d, w3d, 4)[:, :, :3]
            plt.close(fig3d)
            pc_img = np.array(Image.fromarray(pc_img).resize((W, H), Image.LANCZOS))

            # ── compose 4-panel figure ────────────────────────────────────────
            panel_imgs   = [gt_img, rc_img, pc_img, overlay]
            panel_titles = ['GT', 'Reconstructed', 'Cubature Points', 'Overlay']

            fig = plt.figure(figsize=(4 * W / 100, (H / 100) + 1.0), dpi=100)
            fig.patch.set_facecolor(BG)
            gs = GridSpec(2, 4, figure=fig,
                          height_ratios=[H, 80], hspace=0.05, wspace=0.02,
                          left=0.01, right=0.99, top=0.92, bottom=0.01)

            for col, (img, title) in enumerate(zip(panel_imgs, panel_titles)):
                ax = fig.add_subplot(gs[0, col])
                ax.imshow(img)
                ax.set_title(title, fontsize=LABEL_FS, fontweight='bold',
                             color=LABEL_COLOR, pad=5)
                ax.axis('off')

            # annotation bar
            ax_ann = fig.add_subplot(gs[1, :])
            ax_ann.set_facecolor('#eeeeee')
            ax_ann.axis('off')
            ax_ann.text(0.5, 0.5, annot_text,
                        transform=ax_ann.transAxes,
                        ha='center', va='center',
                        fontsize=12, fontweight='bold', color='#111111')

            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            fw, fh = fig.canvas.get_width_height()
            frames.append(buf.reshape(fh, fw, 4)[:, :, :3])
            plt.close(fig)

        imageio.mimsave(out_path, frames, fps=fps, loop=0)
        print(f'Saved point cloud GIF to {out_path}')

    def calculate_loss(self, view_indices=None, start_step=0, end_step=-1):
        images_gt = self.ref[view_indices, start_step:end_step]
        images_pred = torch.stack(self.save_list, dim=1)
        images_pred_loss = images_pred.reshape(-1, 3, images_pred.shape[-2], images_pred.shape[-1])
        images_gt_loss = images_gt.reshape(-1, 3, images_gt.shape[-2], images_gt.shape[-1])
        loss = self.loss_fn(images_pred_loss, images_gt_loss)
        return loss
    
    def calculate_metrics(self, view_indices=None, start_step=0, end_step=-1): 
        images_gt = self.ref[view_indices, start_step:end_step]
        images_pred = torch.stack(self.save_list, dim=1) 
        images_pred_loss = images_pred.reshape(-1, 3, images_pred.shape[-2], images_pred.shape[-1]).clamp(0, 1)
        images_gt_loss = images_gt.reshape(-1, 3, images_gt.shape[-2], images_gt.shape[-1]).clamp(0, 1)
        psnr_list, ssim_list = [], []
        for i in range(images_pred_loss.shape[0]):
            psnr_list.append(self.psnr(images_pred_loss[i:i+1], images_gt_loss[i:i+1]))
            ssim_list.append(self.ssim(images_pred_loss[i:i+1], images_gt_loss[i:i+1]))
        psnr, ssim = torch.stack(psnr_list, dim=0).mean(), torch.stack(ssim_list, dim=0).mean()
        return psnr, ssim
    
    def simulate_step(self):
        self.z_prev = self.z.clone()
        self.z_dot_prev = self.z_dot.clone()
        more_partial_newton_E = partial(self.partial_newton_E, z_prev=self.z_prev, z_dot=self.z_dot)
        more_partial_newton_G = partial(self.partial_newton_G, z_prev=self.z_prev, z_dot=self.z_dot)
        more_partial_newton_H = partial(self.partial_newton_H, z_prev=self.z_prev, z_dot=self.z_dot) 
        self.z = newtons_method(self.z, more_partial_newton_E, more_partial_newton_G,
            more_partial_newton_H, max_iters=self.simulation_newton_iters)
        self.z_dot = ((self.z - self.z_prev) / self.delta_t)

    @torch.no_grad()
    def simulate_fast_forward(self, target_step, view_indices=[0], render=False):
        views = [self.gs_context['gs_views'][idx] for idx in view_indices] if self.args.model_type == 'gs' else None
        for i in range(self.current_step, target_step): 
            if i == 0 and render: 
                self.render_frame(i, views) 
            self.simulate_step()
            if render: 
                self.render_frame(i+1, views) 
        self.current_step = target_step

    def simulate_with_grad(self, target_step, view_indices=[0]): 
        views = [self.gs_context['gs_views'][idx] for idx in view_indices] 
        for i in range(self.current_step, target_step):
            self.simulate_step()
            self.render_frame(i, views)
        self.current_step = target_step
    
    def update_parameters(self, view_indices, start_step, end_step):
        loss = self.calculate_loss(view_indices, start_step, end_step)
        self.mat_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.mat_optimizer.step() 
        return loss