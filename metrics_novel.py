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

from pathlib import Path
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    renders_raw = []
    gts_raw = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if not os.path.splitext(fname)[1] in ['.png', '.jpg']:
            continue
        render = Image.open(os.path.join(renders_dir, fname))
        gt = Image.open(os.path.join(gt_dir, fname))
        renders_raw.append(render)
        gts_raw.append(gt)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, renders_raw, gts_raw, image_names

def evaluate(gt_path, render_path):
    full_dict = {}
    per_view_dict = {}

    renders, gts, _, _, image_names = readImages(render_path, gt_path)

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        gt = gts[idx].squeeze(0).permute(1,2,0).cpu().numpy().reshape(-1, 3)
        render = renders[idx].squeeze(0).permute(1,2,0).cpu().numpy().reshape(-1, 3)
        scale = (gt * render).sum(axis=0) / (render ** 2).sum(axis=0)
        assert scale.shape == (3,), scale.shape
        rgb = (renders[idx] * torch.from_numpy(scale).cuda()[:, None, None]).clamp(0., 1.)

        ssims.append(ssim(rgb, gts[idx]))
        psnrs.append(psnr(rgb, gts[idx]))
        lpipss.append(lpips(rgb, gts[idx], net_type='vgg'))

    full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item()})
    per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
    return full_dict, per_view_dict

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--ignore_envs', nargs="+", type=str, default=[])
    args = parser.parse_args()
    for model_path in args.model_paths:
        print("Scene:", model_path)
        full_dict = {}
        per_view_dict = {}

        full_dict['albedo'] = {}
        per_view_dict['albedo'] = {}
        test_dir = os.path.join(model_path, 'test')
        for method in os.listdir(test_dir):
            print("Albedo Method:", method)
            full, per = evaluate(os.path.join(test_dir, method, 'gt_albedos'), os.path.join(test_dir, method, 'albedos'))
            full_dict['albedo'][method] = full
            per_view_dict['albedo'][method] = per

            print("  SSIM : {:>12.7f}".format(full["SSIM"], ".5"))
            print("  PSNR : {:>12.7f}".format(full["PSNR"], ".5"))
            print("  LPIPS: {:>12.7f}".format(full["LPIPS"], ".5"))
            print("")
        

        full_dict['novel'] = {}
        per_view_dict['novel'] = {}
        novel_dir = os.path.join(model_path, 'novel')
        for method in os.listdir(novel_dir):
            full_dict['novel'][method] = {}
            per_view_dict['novel'][method] = {}
            ssim_s = []
            psnr_s = []
            lpips_s = []
            for hdr_name in os.listdir(os.path.join(novel_dir, method)):
                if hdr_name in args.ignore_envs:
                    continue
                print("Relight Method:", method, "with", hdr_name)
                full, per = evaluate(os.path.join(novel_dir, method, hdr_name, 'gt'), os.path.join(novel_dir, method, hdr_name, 'renders'))
                ssim_s.append(full["SSIM"])
                psnr_s.append(full["PSNR"])
                lpips_s.append(full["LPIPS"])
                full_dict['novel'][method][hdr_name] = full
                per_view_dict['novel'][method][hdr_name] = per
            full_dict['novel'][method]['summary'] = {
                "SSIM": sum(ssim_s) / len(ssim_s), "PSNR": sum(psnr_s) / len(psnr_s), "LPIPS": sum(lpips_s) / len(lpips_s)
            }

            print("  SSIM : {:>12.7f}".format(full_dict['novel'][method]['summary']["SSIM"], ".5"))
            print("  PSNR : {:>12.7f}".format(full_dict['novel'][method]['summary']["PSNR"], ".5"))
            print("  LPIPS: {:>12.7f}".format(full_dict['novel'][method]['summary']["LPIPS"], ".5"))
            print("")
        
        with open(os.path.join(model_path, "novel_results.json"), 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(os.path.join(model_path, "novel_per_view.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)