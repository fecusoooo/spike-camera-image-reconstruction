# -*- coding: utf-8 -*-
# @Time : 2026/6/7
# @Author : Xu Si Nga
# @File : infer_tfi.py

import numpy as np
import torch
import os
import cv2
import nr_iqa
from scipy.stats import kurtosis
from infer_our_data import load_spikes_from_dat, load_spikes_from_open_dat
from spikecv.spkProc.reconstruction.tfi import TFI

if __name__ == '__main__':
    our_data = False
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # ================= 参数 =================
    # dat_path = '/home/lhs183019/snap/snapd-desktop-integration/361/Desktop/BSF/raw_data_our/0.dat'
    dat_path = '/home/lhs183019/snap/snapd-desktop-integration/361/Desktop/BSF/open_data_our/card/spikes.dat'
    output_path = '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/card_data_result/tfi'
    os.makedirs(output_path, exist_ok=True)
    output_video = f'{output_path}/recon_tfi.mp4'
    fps = 30
    step = 3
    max_half_win = 20

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

    # ================= TFI 重建 =================
    print("运行 TFI ...")
    reconstructor = TFI(spike_h=H, spike_w=W, device=device)
    rec_frames = reconstructor.spikes2images(spikes_full, max_search_half_window=max_half_win)
    T_rec = rec_frames.shape[0]
    print(f"重建帧数: {T_rec}")

    # ================= 评估帧选取：与 BSF 时间戳对齐 =================
    eval_start = 30
    eval_indices = range(eval_start, T_rec, step)

    # ================= 写视频 & 计算 IQA =================
    iqa_scores = {'BRISQUE': [], 'PIQE': [], 'NIQE': []}
    lap_tfi = []
    mad_list = []
    kurt_list = []
    ratio_list = []
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    
    if our_data:
        out = cv2.VideoWriter(output_video, fourcc, fps, (W, H), isColor=False)
    else:
        out = cv2.VideoWriter(output_video, fourcc, fps, (W_raw, H_raw), isColor=False)

    for cnt,idx in enumerate(eval_indices):
        img_linear = rec_frames[idx]

        img_gamma = ((img_linear.astype(np.float32) / 255.0) ** (1/2.2) * 255).astype(np.uint8)

        if not our_data:
            img_gamma = img_gamma[:H_raw, :W_raw]

        out.write(img_gamma)
        if cnt == 0 or cnt == 10:
            cv2.imwrite(f'{output_path}/frame_{cnt:04d}.png', img_gamma)
        lap_tfi.append(cv2.Laplacian(img_gamma, cv2.CV_64F).var())
            
        img_median = cv2.medianBlur(img_gamma, 3)
        residual = img_gamma.astype(np.float32) - img_median.astype(np.float32)
        mad = np.mean(np.abs(residual))
        kurt = kurtosis(residual.flatten(), fisher=False)
        ratio = nr_iqa.high_freq_energy_ratio(img_gamma)
        mad_list.append(mad)
        kurt_list.append(kurt)
        ratio_list.append(ratio)

        scores = nr_iqa.compute_all(img_gamma)
        for k, v in scores.items():
            iqa_scores[k].append(v)
        print(f"Frame {idx:4d} (TS={max_half_win+idx}): "
                f"BRISQUE={scores['BRISQUE']:.3f}  PIQE={scores['PIQE']:.3f}  NIQE={scores['NIQE']:.3f}")

    out.release()

    # ================= 统计与保存 =================
    print("\n========== TFI NR-IQA 评估结果 ==========")
    print(f"平均 Laplacian 变异数: {np.mean(lap_tfi):.2f}")

    np.save(f"{output_path}/laplacian_var_tfi.npy", np.array(lap_tfi))
    for metric, values in iqa_scores.items():
        if values:
            arr = np.array(values)
            print(f"{metric:8s}: mean={arr.mean():.4f} ± {arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
            np.save(f"{output_path}/iqa_scores_{metric}.npy", arr)

    print("\n========== TFI 中值残差指标 ==========")
    print(f"残差 MAD  (均值±标准差): {np.mean(mad_list):.4f} ± {np.std(mad_list):.4f}")
    print(f"残差峰度 (均值±标准差): {np.mean(kurt_list):.4f} ± {np.std(kurt_list):.4f}")
    print(f"频域高频能量比 (均值±标准差): {np.mean(ratio_list):.4f} ± {np.std(ratio_list):.4f}")
    np.save(f"{output_path}/mad_tfi.npy", np.array(mad_list))
    np.save(f"{output_path}/kurtosis_tfi.npy", np.array(kurt_list))
    np.save(f"{output_path}/high_freq_energy_ratio_tfi.npy", np.array(ratio_list))

    print("完成！")