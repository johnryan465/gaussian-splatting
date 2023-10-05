#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from scene.util import se3_exp_map
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_eval
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.profiler import profile, record_function, ProfilerActivity

from utils.se3 import random_rotation

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard available: logging progress")
except ImportError:
    TENSORBOARD_FOUND = False

import copy

def finite_difference_gradient(f, x, eps=5e-6):
    return ((f(x + eps/2) - f(x - eps/2)) / eps)


def finite_diff_calc(i, gt_image, viewpoint_cam, gaussians, pipe, background, opt):
    def f(x):
        random_offset_tensor = torch.zeros(6, device="cuda")
        random_offset_tensor[i] = x
        mul = se3_exp_map(random_offset_tensor.view(1, 6)).view(4, 4)
        # print("Mul", mul)
        cam = copy.deepcopy(viewpoint_cam)
        cam.world_view_transform = cam.world_view_transform.detach().clone() @ mul
        render_pkg = render(cam, gaussians, pipe, background)
        # tmp = viewpoint_cam.world_view_transform.cpu().detach().numpy()
        image = render_pkg["render"]

        # Loss
        # gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        return loss.item()
    return finite_difference_gradient(f, 0.0)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        print("Loading checkpoint from", checkpoint)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viewpoint_stack = scene.getTrainCameras()
    original_pos = None
    offset = None
    viewpoint_stack_idxs_ = [] # = [ i for i in range(len(viewpoint_stack))]

    for i, cam in enumerate(viewpoint_stack):
        random_offset = (torch.rand(3, device="cuda") * 1) - 0.5
        random_offset_tensor = torch.zeros(4, 4, device="cuda")
        # random_offset_tensor[3, :3] = random_offset

        if cam.image_name == "000089":
            viewpoint_stack_idxs_.append(i)
            original_pos = cam.camera_center.clone()
            print("Camera Transformation", cam.world_view_transform)
            print("Camera Transformation Inverse", torch.inverse(cam.world_view_transform))
            cam.world_view_transform =  torch.inverse(torch.inverse(torch.tensor(random_rotation(20)).float().cuda() @ (cam.world_view_transform) + random_offset_tensor))
            print("Original pos", original_pos)
            print("Moved pos", cam.camera_center)
            offset = random_offset
        # print(cam.camera_center)
    #if iteration == 2500:

    viewpoint_stack_idxs = viewpoint_stack_idxs_.copy()
    gaussians.enable_training_camera(viewpoint_stack[viewpoint_stack_idxs[0]])
    gaussians.disable_params()
            
    for iteration in range(first_iter, opt.iterations + 1):
        # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:     
        if True:   
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration - first_iter)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if len(viewpoint_stack_idxs) == 0:
                viewpoint_stack_idxs = viewpoint_stack_idxs_.copy()
            viewpoint_cam_idx = viewpoint_stack_idxs.pop(randint(0, len(viewpoint_stack_idxs)-1))
            viewpoint_cam = viewpoint_stack[viewpoint_cam_idx]
            viewpoint_cam.set_iteration(iteration)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            # tmp = viewpoint_cam.world_view_transform.cpu().detach().numpy()
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            # print("Leaf", gaussians.world_view_transform.is_leaf)
            # print("Val", gaussians.world_view_transform.grad)
            # print("Grad", gaussians.camera_center.grad)

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_eval, (pipe, background), viewpoint_stack)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                # Densification
                #if iteration < opt.densify_until_iter:
                #    # Keep track of max radii in image-space for pruning
                #    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                #    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #        gaussians.reset_opacity()

                # print(gaussians.world_view_transform.grad)
                # print(gaussians.exp_factor)
                # print(gaussians.camera_center)
                # print("Grad", gaussians._omega.grad)
                #print("Omega", gaussians._omega)
                #print(gaussians.world_view_transform)
                # Optimizer step
                viewpoint_cam.world_view_transform = gaussians.world_view_transform.clone()
                #print(gaussians.world_view_transform)
                # print(gaussians.camera_center)
                # print(
                #     (gaussians.world_view_transform - 0.001 * gaussians.world_view_transform.grad).inverse()[3,:3]
                # )

                #if viewpoint_cam.image_name == "000064":
                #    print("Grad", gaussians._world_view_transform_inv.grad[3, :3])
                #print("Grad", gaussians._omega.grad)
                #print("Leaf", gaussians._omega.is_leaf)
                #print("Analytical", gaussians._omega.grad)

                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                #print(gaussians.world_view_transform)
                # print(viewpoint_cam.world_view_transform)
                # print(viewpoint_cam.world_view_transform - gaussians.world_view_transform.detach().clone())
                # viewpoint_cam.world_view_transform = gaussians.world_view_transform.detach().clone()
                #print("Finite differences", finite_diff_calc(0, gt_image, viewpoint_cam, gaussians, pipe, background, opt))
                #print("Finite differences", finite_diff_calc(1, gt_image, viewpoint_cam, gaussians, pipe, background, opt))
                #print("Finite differences", finite_diff_calc(2, gt_image, viewpoint_cam, gaussians, pipe, background, opt))
                #print("Finite differences", finite_diff_calc(3, gt_image, viewpoint_cam, gaussians, pipe, background, opt))
                #print("Finite differences", finite_diff_calc(4, gt_image, viewpoint_cam, gaussians, pipe, background, opt))
                #print("Finite differences", finite_diff_calc(5, gt_image, viewpoint_cam, gaussians, pipe, background, opt))

                # viewpoint_cam._omega = gaussians._omega.detach().clone()
                # print(gaussians.exp_factor)
                # viewpoint_cam.world_view_transform = gaussians.world_view_transform.detach().clone()

                if viewpoint_cam.image_name == "000089":
                    #print("Original pos", original_pos)
                    # print("Offset", offset)
                    #print("New pos", viewpoint_cam.camera_center)
                    tb_writer.add_scalar('diff', torch.norm(viewpoint_cam.camera_center - original_pos).cpu().numpy(), iteration)
                    #print("Diff", torch.norm(viewpoint_cam.camera_center - original_pos))
                    #print("Diff Tensor", viewpoint_cam.camera_center - original_pos)
                
                # if iteration == 2500:
                #    gaussians.enable_training_camera()
                # if iteration % 700 == 0:
                torch.cuda.empty_cache()
                
                # print("Current:", torch.cuda.memory_allocated()) #  / torch.cuda.max_memory_allocated())
                # print("Peak:", torch.cuda.max_memory_allocated())



                

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    print("Saving checkpoint to", scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total"))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_cameras):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = [{'name': 'train', 'cameras' : [train_cameras[idx % len(train_cameras)] for idx in range(5, 30, 5)]}]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_out["render"], 0.0, 1.0)
                    print("Image", image.shape)
                    depth = render_out["depth"]
                    # print(torch.max(depth))
                    # scale depth to 0-1
                    depth = (depth) / (torch.max(depth))
                    # convert to 3 channel image for tensorboard
                    depth = depth.repeat(3, 1, 1)
                    

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[0, 100, 200, 500, 700, 1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 7_001, 7_050, 7_100, 7_200, 7_300, 7_400, 7_500, 7_700, 8_000, 9_000, 10_000, 11_000, 12_000, 13_000, 14_000, 15_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
