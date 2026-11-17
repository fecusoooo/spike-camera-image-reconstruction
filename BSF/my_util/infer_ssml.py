# -*- coding: utf-8 -*- 
# @Time : 2026/6/7
# @Author : Xu Si Nga
# @File : infer_ssml.py

import numpy as np
import torch
import sys
import os
from collections import OrderedDict
import cv2
import nr_iqa

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append('..')
from spikecv.spkProc.reconstruction.SSML_Recon.ssml_model import SSML_ReconNet
from infer_our_data import load_spikes_from_dat, load_spikes_from_open_dat

if __name__ == '__main__':
    our_data = False
    device = torch.device('cuda:3')

    # ================= 参数 =================
    # dat_path = '/home/lhs183019/snap/snapd-desktop-integration/361/Desktop/BSF/raw_data_our/0.dat'
    dat_path = '/home/lhs183019/snap/snapd-desktop-integration/361/Desktop/BSF/open_data_our/card/spikes.dat'
    output_path = '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/card_data_result/ssml'
    os.makedirs(output_path, exist_ok=True)
    output_video = f'{output_path}/recon_video_gamma_0.mp4'
    fps = 30
    step = 3
    max_half_win = 20
    window_size = 41

    eval_freq = 1
    save_iqa_path = 'iqa_scores.npz'

    # ================= 设备与模型 =================
    model = SSML_ReconNet()
    model_path = "/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/SpikeCV/SpikeCV/spkProc/reconstruction/SSML_Recon/pretrained/pretrained_ssml_recon.pt"
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)

    # ================= 数据准备 =================
    print("解析 .dat 并准备数据...")
    if our_data:
        spikes_full = load_spikes_from_dat(dat_path)          # (T,1000,1000) numpy
    else:
        spikes_full_raw = load_spikes_from_open_dat(dat_path, 400, 400, 250)
        H_raw, W_raw = spikes_full_raw.shape[1], spikes_full_raw.shape[2]  # 250, 400
        target_H = ((H_raw + 15) // 16) * 16
        target_W = ((W_raw + 15) // 16) * 16
        pad_bottom = target_H - H_raw
        pad_right = target_W - W_raw
        spikes_full = np.pad(spikes_full_raw, ((0,0), (0,pad_bottom), (0,pad_right)), mode='constant', constant_values=0)

    start = max_half_win
    end = spikes_full.shape[0] - max_half_win
    spikes_aligned = spikes_full[start:end]
    T_valid = spikes_aligned.shape[0]
    print(f"有效帧总数: {T_valid}")

    del spikes_full
    torch.cuda.empty_cache()

    # ================= 滑动窗口重建视频 =================
    half_win = window_size // 2
    bsf_half_win = 30
    center_start = bsf_half_win
    center_end = T_valid - bsf_half_win

    frame_list = []
    iqa_scores = {'BRISQUE': [], 'PIQE': [], 'NIQE': []}
    model.eval()
    lap_ssml = []
    
    for idx, center in enumerate(range(center_start, center_end, step)):
        t_start = center - half_win
        t_end   = center + half_win + 1

        spike_block = spikes_aligned[t_start:t_end]

        spike_t = torch.from_numpy(spike_block).float().unsqueeze(0).to(device)
        with torch.no_grad():
            res = model(spike_t, train=False)
        img = res[0, 0].cpu().numpy()

        img = np.clip(img, 0, 1)
        img_gamma = img ** (1/2.2) * 255.0
        img_gamma = img_gamma.astype(np.uint8)
        
        if not our_data:
            img_gamma = img_gamma[:H_raw, :W_raw]
            
        frame_list.append(img_gamma)
        lap_ssml.append(cv2.Laplacian(img_gamma, cv2.CV_64F).var())
        print(f"Laplacian var SSML: {lap_ssml[-1]:.2f}")

        if idx % eval_freq == 0:
            scores = nr_iqa.compute_all(img_gamma)
            for metric_name, score_val in scores.items():
                iqa_scores[metric_name].append(score_val)
            print(f"Frame {idx:4d}: BRISQUE={scores['BRISQUE']:.3f}  PIQE={scores['PIQE']:.3f}  NIQE={scores['NIQE']:.3f}")

        del spike_t, res    
        torch.cuda.empty_cache()

        print(f"重建进度: {idx+1}/{ (center_end - center_start)//step + 1}")

    # ================= 保存视频 =================
    print(f"保存视频到 {output_video} ...")
    h, w = frame_list[0].shape
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h), isColor=False)
    for frame in frame_list:
        out.write(frame)
    out.release()

    # ================= 输出 NR-IQA 统计结果 =================
    print("\n========== NR-IQA 评估结果 ==========")
    for metric, values in iqa_scores.items():
        if values:
            arr = np.array(values)
            print(f"{metric:8s}: mean={arr.mean():.4f} ± {arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
        else:
            print(f"{metric:8s}: 无数据")

    print(f"平均 Laplacian 变异数: {np.mean(lap_ssml):.2f}")
    np.save(f"{output_path}/laplacian_var_ssml.npy", np.array(lap_ssml))

    if save_iqa_path and any(iqa_scores.values()):
        base = os.path.splitext(save_iqa_path)[0]
        for metric, values in iqa_scores.items():
            if values:
                np.save(f"{output_path}/{base}_{metric}.npy", np.array(values))
        print(f"指标详细数据已保存到 {output_path}/{base}_BRISQUE.npy, {output_path}/{base}_PIQE.npy, {output_path}/{base}_NIQE.npy")

    print("完成！")