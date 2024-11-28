import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, cos_loss, bce_loss, knn_smooth_loss
from gaussian_renderer import render, network_gui
import numpy as np
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from datetime import datetime as dt
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, match_depth, normal2curv, resize_image, cross_sample
from utils.transi_utils import batch_histogram, normalize_hist, total_variation_loss, convolve_histograms, image_snr,snr_shifting
from torchvision.utils import save_image
from argparse import ArgumentParser, Namespace
import time
import os
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import ot

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, opt.camera_lr, shuffle=False, resolution_scales=[1, 2, 5])
    use_mask = dataset.use_mask
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif use_mask: # visual hull init
        gaussians.mask_prune(scene.getTrainCameras(), 4)
        None

    opt.densification_interval = max(opt.densification_interval, len(scene.getTrainCameras()))

    if opt.only_use == "rgb":
        opt.transi_only_until = 0
    if opt.only_use == "lidar": 
        opt.transi_only_until = 20000

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    calibration_histogram = np.load('./sampling/flat_wall.npy')
    pulse = np.mean(calibration_histogram[:, 0, :], axis=0)
    pulse = torch.tensor(pulse, dtype=torch.float32, device="cuda")
    pulse /= pulse.sum()
    max_index = torch.argmax(pulse)
    shift_amount = 128 - max_index
    start_index = max(0, shift_amount)
    end_index = min(256, shift_amount + pulse.shape[0])
    pulse_start = max(0, -shift_amount)
    pulse_end = pulse_start + (end_index - start_index)
    extended_pulse = torch.zeros(256, device="cuda")
    extended_pulse[start_index:end_index] = pulse[pulse_start:pulse_end]
    pulse = extended_pulse

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    pool = torch.nn.MaxPool2d(9, stride=1, padding=4)

    viewpoint_stack = None
    viewpoint_stack_fullsize = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    count = -1
    for iteration in range(first_iter, opt.iterations + 2):

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if iteration - 1 == 0:
            scale = 5
        elif iteration - 1 == 1000 + opt.transi_only_until:
            scale = 2
        elif iteration - 1 == 2000 + opt.transi_only_until:
            scale = 1
        if opt.scale_single == 1:
            scale = 1

        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale).copy()[:]
            viewpoint_stack_fullsize = scene.getTrainCameras(1).copy()[:]
        randcam = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam_fullsize = viewpoint_stack_fullsize.pop(randcam)
        viewpoint_cam = viewpoint_stack.pop(randcam)
        
        # render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        background = torch.rand((3), dtype=torch.float32, device="cuda") if dataset.random_background else background
        patch_size = [float('inf'), float('inf')]

        # coarse to fine sampling 
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, patch_size)
        image, normal, depth, _, opac, _, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["depth_buffer"], render_pkg["opac"], render_pkg["opac_buffer"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # fullsize sampling for transient
        render_pkg_fullsize = render(viewpoint_cam_fullsize, gaussians, pipe, background, patch_size)
        depth_buffer_nomask, opac_buffer_nomask, opac_fullsize = render_pkg_fullsize["depth_buffer"], render_pkg_fullsize["opac_buffer"], render_pkg_fullsize["opac"]
        
        mask_gt = viewpoint_cam.get_gtMask(use_mask)
        gt_image = viewpoint_cam.get_gtImage(background, use_mask)
        gt_image_fullsize = viewpoint_cam_fullsize.get_gtImage(background, use_mask)
        _, gt_im_H, gt_im_W = gt_image.shape
        gt_transi = viewpoint_cam.get_gtTransi() if viewpoint_cam.get_gtTransi() is not None else None
        if gt_transi is None: 
            breakpoint() 
        mask_vis = (opac.detach() > 1e-5)
        mask_vis_fullsize = (opac_fullsize.detach() > 1e-5)
        normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
        d2n = depth2normal(depth, mask_vis, viewpoint_cam)
        mono = viewpoint_cam.mono if dataset.mono_normal else None
        if mono is not None:
            mono *= mask_gt
            monoN = mono[:3]


        # ===== transient work =====

        if gt_transi is not None and depth_buffer_nomask is not None and opac_buffer_nomask is not None and opt.only_use != "rgb": 

            depth_buffer = depth_buffer_nomask * mask_vis_fullsize
            opac_buffer = opac_buffer_nomask * mask_vis_fullsize

            depth_histogram = batch_histogram(depth_buffer, opac_buffer, opt.hist_near, opt.hist_far, opt.num_hist_bins)

            transi_bins, H_pred, W_pred = depth_histogram.shape
            H_gt, W_gt = gt_transi.shape[1], gt_transi.shape[2]
            scale_H = H_pred // H_gt
            scale_W = W_pred // W_gt

            depth_histogram_downscaled = depth_histogram.view(
                transi_bins, H_gt, scale_H, W_gt, scale_W
            ).sum(dim=(2, 4)) 

            convolved_histograms = convolve_histograms(depth_histogram_downscaled, pulse)

            normalized_depth_histogram = normalize_hist(convolved_histograms)
            normalized_gt_transi = normalize_hist(gt_transi)
            normalized_depth_histogram_log = torch.log(normalized_depth_histogram)

            strd = opt.strd
            mean = torch.nn.functional.avg_pool2d(gt_image, strd, stride=strd)
            mean_sq = torch.nn.functional.avg_pool2d(gt_image ** 2, strd, stride=strd)
            variance = (mean_sq - mean ** 2).mean(dim=0)

            def scene_adaptive_sigmoid(x, steepness=10000, offset=0.001):
                return torch.clamp(1 / (1 + torch.exp(-steepness * (x - offset))), 0.00001, 1.0)
            
            img_snr = image_snr(gt_image_fullsize)
            snr_offset = snr_shifting(img_snr, opt.snr_scale_a, opt.snr_scale_b)
            offset = snr_offset 
            # uncomment to manually set sigmoid offset 
            # offset = opt.sigmoid_offset
            
            colorvar_weights_downscaled = scene_adaptive_sigmoid(variance, steepness=opt.sigmoid_steepness, offset=offset)

            colorvar_weights_upscaled = torch.nn.functional.interpolate(colorvar_weights_downscaled.unsqueeze(0).unsqueeze(0),
                                                                size=(gt_im_H, gt_im_W),
                                                                mode='nearest').squeeze(0).squeeze(0) 
                                                                
            transi_weights = 1 - colorvar_weights_downscaled 
            transi_loss_full = torch.nn.functional.kl_div(normalized_depth_histogram_log, normalized_gt_transi, reduction='sum')

            if iteration < opt.transi_only_until:
                transi_loss = (transi_loss_full/normalized_depth_histogram_log.size(0)).mean()
            else: 
                transi_loss = ((transi_loss_full/normalized_depth_histogram_log.size(0)) * transi_weights).mean() 

            
            if opt.lambda_tv != 0: 
                tv_loss = total_variation_loss(depth)
                transi_loss += opt.lambda_tv * tv_loss 
        else: 
            transi_loss = 0

        # print radii sanity check 
        # if iteration % 100 == 0 : print(radii.max())

        # ==========
                
        # Loss
        if iteration > opt.transi_only_until: 
            if iteration % opt.intersperse_rgb == 0:
                loss_kwargs = {}
            else: 
                loss_kwargs = {'weight': colorvar_weights_upscaled.unsqueeze(0)} if 'colorvar_weights_upscaled' in locals() and (gt_transi is not None and opt.only_use != "rgb") else {}
        else: 
            loss_kwargs = {}

        Ll1 = l1_loss(image, gt_image, **loss_kwargs) 
        ssim_weight = colorvar_weights_upscaled.mean()
        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image) * ssim_weight)
        
        loss_mask = (opac * (1 - pool(mask_gt))).mean()
        
        if mono is not None:
            loss_monoN = cos_loss(normal, monoN, weight=mask_gt)
        loss_surface = cos_loss(normal, d2n)
        
        opac_ = gaussians.get_opacity
        opac_mask0 = torch.gt(opac_, 0.01) * torch.le(opac_, 0.5)
        opac_mask1 = torch.gt(opac_, 0.5) * torch.le(opac_, 0.99)
        opac_mask = opac_mask0 * 0.01 + opac_mask1
        loss_opac = (torch.exp(-(opac_ - 0.5)**2 * 20) * opac_mask).mean()
        
        curv_n = normal2curv(normal, mask_vis)
        loss_curv = l1_loss(curv_n * 1, 0) 
        
        if opt.only_use == "rgb":
            loss = loss_rgb
        elif opt.only_use == "lidar" and gt_transi is not None: 
            loss = opt.transi_weight * transi_loss
        elif opt.only_use == "lidar" and gt_transi is None: 
            raise Exception("Can't use lidar only if no gt transients available!")
        else:
            if gt_transi is not None:
                if iteration < opt.transi_only_until:
                    loss = opt.transi_weight * transi_loss
                else:
                    loss = transi_loss
                    loss += loss_rgb * (1/opt.transi_weight)
            else:
                loss = loss_rgb  # user didn't specify rgb only but no transi exists 

        loss += 0.1 * loss_mask
        loss += (0.01 + 0.1 * min(2 * iteration / opt.iterations, 1)) * loss_surface
        loss += 0.005 * loss_curv
        loss += 0.01* loss_opac
        if mono is not None:
            loss += (0.04 - ((iteration / opt.iterations)) * 0.02) * loss_monoN

        loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_rgb.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}, Pts={len(gaussians._xyz)}"}
                if not isinstance(transi_loss, int):
                    postfix["TransiLoss"] = f"{transi_loss.item()}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # log and save
            test_background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, pipe, test_background, use_mask)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # prune settings
            cull_over = opt.cull_over
            cull_every = opt.cull_every

            if iteration % cull_every == 0 and iteration < opt.transi_only_until:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                pruned_data = gaussians.prune_large_gaussians(cull_over / opt.cull_over_transi_only)
                gaussians.densify_around_pruned_points(*pruned_data, scene.cameras_extent)

            # densification 
            if iteration > opt.transi_only_until:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                min_opac = 0.1
                if iteration % opt.densification_interval == 0:
                    gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                    gaussians.adaptive_densify(opt.densify_grad_threshold, scene.cameras_extent)
                if iteration % cull_every == 0:
                    pruned_data = gaussians.prune_large_gaussians(cull_over)
                    gaussians.densify_around_pruned_points(*pruned_data, scene.cameras_extent, N=opt.prune_replace)
                if (iteration-1) % opt.opacity_reset_interval == 0 and opt.opacity_lr > 0:
                    gaussians.reset_opacity(0.12, iteration)


            if iteration > 1 and (iteration - 1) % opt.train_viz_update == 0: #4999 == 0:
                print("Saving to test folder")

                normal_wrt = normal2rgb(normal, mask_vis)
                depth_wrt = depth2rgb(depth, mask_vis)
                img_wrt = torch.cat([gt_image, image, normal_wrt * opac, depth_wrt * opac], 2)
                save_image(img_wrt.cpu(), 'test/train.png') 
            
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad() 

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):       
    data_str = args.source_path.split('/')[-1]
    time_str = dt.now().strftime('day%m%d_time%H%M')
    if not args.model_path:
        args.model_path = os.path.join("./output", f"{data_str}_{time_str}")
    else: 
        args.model_path = os.path.join("./output", args.model_path)
        
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, pipe, bg, use_mask):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()[::8]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(render(viewpoint, scene.gaussians, pipe, bg, [float('inf'), float('inf')])["render"], 0.0, 1.0)
                    depth = torch.clamp(render(viewpoint, scene.gaussians, pipe, bg, [float('inf'), float('inf')])["depth"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.get_gtImage(bg, with_mask=use_mask), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
