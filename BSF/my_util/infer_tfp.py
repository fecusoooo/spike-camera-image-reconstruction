# -*- coding: utf-8 -*-
# @Time : 2026/6/7
# @Author : Xu Si Nga
# @File : infer_tfp.py

import numpy as np
import torch
import cv2
import nr_iqa
import os
from scipy.stats import kurtosis
from infer_our_data import load_spikes_from_dat, load_spikes_from_open_dat
from spikecv.spkProc.reconstruction.tfp import TFP


if __name__ == '__main__':
    our_data = False
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # ================= 参数 =================
    # dat_path = '/home/lhs183019/snap/snapd-desktop-integration/361/Desktop/BSF/raw_data_our/0.dat'
    dat_path = '/home/lhs183019/snap/snapd-desktop-integration/361/Desktop/BSF/open_data_our/card/spikes.dat'
    output_path = '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/card_data_result/tfp'
    os.makedirs(output_path, exist_ok=True)
    output_video = f'{output_path}/recon_tfp_gamma.mp4'
    fps = 30
    step = 3
    half_win_length = 20

    # ================= 数据加载 =================
    print("解析 .dat ...")
    if our_data:
        spikes_full = load_spikes_from_dat(dat_path)   # (T, 1000, 1000)
        H, W = spikes_full.shape[1], spikes_full.shape[2]  # 1000, 1000
    else:
        spikes_full_raw = load_spikes_from_open_dat(dat_path, 400, 400, 250)
        H_raw, W_raw = spikes_full_raw.shape[1], spikes_full_raw.shape[2]  # 250, 400
        target_H = ((H_raw + 15) // 16) * 16
        target_W = ((W_raw + 15) // 16) * 16
        pad_bottom = target_H - H_raw
        pad_right = target_W - W_raw
        spikes_full = np.pad(spikes_full_raw, ((0,0), (0,pad_bottom), (0,pad_right)), mode='constant', constant_values=0)
        H, W = target_H, target_W

    # ================= TFP 重建 =================
    print("运行 TFP ...")
    reconstructor = TFP(spike_h=H, spike_w=W, device=device)
    rec_frames = reconstructor.spikes2images(spikes_full, half_win_length=half_win_length)
    T_rec = rec_frames.shape[0]
    print(f"重建帧数: {T_rec}")

    # ================= 采样 & 评估（对齐 BSF 的时间戳） =================
    iqa_scores = {'BRISQUE': [], 'PIQE': [], 'NIQE': []}
    mad_list = []
    kurt_list = []
    ratio_list = []
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    
    if our_data:
        out = cv2.VideoWriter(output_video, fourcc, fps, (W, H), isColor=False)
    else:
        out = cv2.VideoWriter(output_video, fourcc, fps, (W_raw, H_raw), isColor=False)
    
    lap_tfp = []
    start_frame = 50 - half_win_length
    for cnt,idx in enumerate(range(start_frame, T_rec, step)):
        img_linear = rec_frames[idx]
        img_float = img_linear.astype(np.float32) / 255.0
        img_gamma = (np.clip(img_float, 0, 1) ** (1/2.2) * 255).astype(np.uint8)
        
        if not our_data:
            img_gamma = img_gamma[:H_raw, :W_raw]
        
        lap_tfp.append(cv2.Laplacian(img_gamma, cv2.CV_64F).var())
            
        img_median = cv2.medianBlur(img_gamma, 3)
        residual = img_gamma.astype(np.float32) - img_median.astype(np.float32)
        mad = np.mean(np.abs(residual))
        kurt = kurtosis(residual.flatten(), fisher=False)
        ratio = nr_iqa.high_freq_energy_ratio(img_gamma)
        mad_list.append(mad)
        kurt_list.append(kurt)
        ratio_list.append(ratio)
        
        if cnt == 0 or cnt == 10:
            cv2.imwrite(f'{output_path}/frame_{cnt:04d}.png', img_gamma)
        out.write(img_gamma)
        scores = nr_iqa.compute_all(img_gamma)
        for k, v in scores.items():
            iqa_scores[k].append(v)
        print(f"Frame {idx:4d}: BRISQUE={scores['BRISQUE']:.3f}  PIQE={scores['PIQE']:.3f}  NIQE={scores['NIQE']:.3f}")

    out.release()

    # ================= 输出统计 =================
    print("\n========== TFP NR-IQA 评估结果 (Gamma 2.2) ==========")
    print(f"平均 Laplacian 变异数: {np.mean(lap_tfp):.2f}")
    np.save(f"{output_path}/laplacian_var_tfp.npy", np.array(lap_tfp))
    for metric, values in iqa_scores.items():
        if values:
            arr = np.array(values)
            print(f"{metric:8s}: mean={arr.mean():.4f} ± {arr.std():.4f}")
            np.save(f"{output_path}/iqa_tfp_{metric}.npy", arr)

    print("\n========== TFP 中值残差指标 ==========")
    print(f"残差 MAD  (均值±标准差): {np.mean(mad_list):.4f} ± {np.std(mad_list):.4f}")
    print(f"残差峰度 (均值±标准差): {np.mean(kurt_list):.4f} ± {np.std(kurt_list):.4f}")
    print(f"频域高频能量比 (均值±标准差): {np.mean(ratio_list):.4f} ± {np.std(ratio_list):.4f}")
    np.save(f"{output_path}/mad_tfp.npy", np.array(mad_list))
    np.save(f"{output_path}/kurtosis_tfp.npy", np.array(kurt_list))
    np.save(f"{output_path}/high_freq_energy_ratio_tfp.npy", np.array(ratio_list))

    print("完成！")