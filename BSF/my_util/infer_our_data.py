# -*- coding: utf-8 -*- 
# @Time : 2026/6/17
# @Author : Xu Si Nga
# @File : infer_our_data.py

import numpy as np
import torch
import sys
import os
from collections import OrderedDict
from scipy.stats import kurtosis


# 关键：防止显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append('..')
from prepare_data.DSFT import DSFT
from models.bsf.dsft_convert import convert_dsft4
from models.bsf.bsf import BSF
import cv2
import nr_iqa # for computing IQA scores on reconstructed frames

# ================= 纯 numpy 解析函数 =================
# def load_spikes_from_dat(dat_path, decode_width=1024, target_width=1000, height=1000):
#     raw_data = np.fromfile(dat_path, dtype=np.uint8)
#     frame_bytes = height * decode_width // 8
#     total_frames = len(raw_data) // frame_bytes
#     raw_data = raw_data[:total_frames * frame_bytes]
#     raw_data = raw_data.reshape(total_frames, height, decode_width // 8)
#     bits = np.unpackbits(raw_data, axis=-1, bitorder='little')
#     spikes_raw = bits.reshape(total_frames, height, decode_width)
#     # delete_cols = np.hstack([np.arange(500, 512), np.arange(1000, 1012)])
#     delete_cols = np.hstack([np.arange(500, 512), np.arange(1000, 1024)])
#     spikes = np.delete(spikes_raw, delete_cols, axis=2)
#     # return spikes_raw.astype(np.float32)
#     return spikes.astype(np.float32)

def load_spikes_from_open_dat(dat_path, decode_width=1024, target_width=1000, height=1000):
    raw_data = np.fromfile(dat_path, dtype=np.uint8)
    frame_bytes = height * decode_width // 8
    total_frames = len(raw_data) // frame_bytes
    raw_data = raw_data[:total_frames * frame_bytes]
    raw_data = raw_data.reshape(total_frames, height, decode_width // 8)
    bits = np.unpackbits(raw_data, axis=-1, bitorder='little')
    spikes_raw = bits.reshape(total_frames, height, decode_width)
    # spikes_raw = np.flip(spikes_raw, axis=2).copy()  # axis=2 对应列方向
    spikes_raw = np.flip(spikes_raw, axis=1).copy()   # 垂直翻转

    return spikes_raw.astype(np.float32)

def load_spikes_from_dat(dat_path, decode_width=1024, target_width=1000, height=1000):
    frame_pixel_bytes = height * (decode_width // 8)   # 128000

    with open(dat_path, 'rb') as f:
        raw_data = f.read()

    # 1. 去掉第一帧前面的 64 字节帧头
    pixel_data = raw_data[64:]
    total_frames = len(pixel_data) // frame_pixel_bytes
    if total_frames == 0:
        raise ValueError("No complete frames found")

    # 2. 将原始二进制数据解码为脉冲帧序列
    pixel_data = pixel_data[:total_frames * frame_pixel_bytes]
    pixels = np.frombuffer(pixel_data, dtype=np.uint8).reshape(total_frames, frame_pixel_bytes) # 转换为 numpy 数组
    pixels_reshaped = pixels.reshape(total_frames, height, decode_width // 8) # reshape 为 (T, 1000, 128)
    bits = np.unpackbits(pixels_reshaped, axis=-1, bitorder='little')  # 位展开 (little-endian), (T, 1000, 1024)

    # 3. 修正删除无效列的索引
    # 一行 1024 列由两个 512 宽度的 block 组成。
    # 左半区有效数据是 0~499，无效列是 500~511。
    # 右半区有效数据是 512~1011，无效列是 1012~1023。
    delete_cols = np.hstack([np.arange(500, 512), np.arange(1012, 1024)])
    spikes = np.delete(bits, delete_cols, axis=2)   # 得到纯净的 (T, 1000, 1000)

    # 4. 调整象限错位
    # 将图像在高度和宽度方向上各自平移，把画面四角的“中心点”拼回到真正的中心
    spikes = np.roll(spikes, shift=(height // 3 + height // 20, target_width // 2), axis=(1, 2))

    # 5. 镜像翻转修复
    spikes = np.flip(spikes, axis=1).copy()   # 垂直翻转
    
    return spikes.astype(np.float32)

if __name__ == '__main__':

    our_data = False
    device = torch.device('cuda:3')

    # ================= 参数 =================
    # dat_path = '/home/lhs183019/snap/snapd-desktop-integration/361/Desktop/BSF/raw_data_our/0.dat'
    dat_path = '/home/lhs183019/snap/snapd-desktop-integration/361/Desktop/BSF/open_data_our/rot/spikes.dat'
    output_path = '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/rot_data_result/bsf'
    output_video = f'{output_path}/recon_video_gamma_0.mp4'
    weight_path = '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/ckpt/bsf.pth'
    fps = 30
    step = 3
    max_half_win = 20
    window_size = 61
    print("解析 .dat 并准备数据...")
    if our_data:
        # ================= 数据准备 =================
        spikes_full = load_spikes_from_dat(dat_path)          # (T,1000,988) numpy
        print("计算 DSFT（GPU）...")
        dsft_calc = DSFT(spike_h=1000, spike_w=1000, device=device)
        dsft_full = dsft_calc.spikes2images(spikes_full, max_search_half_window=max_half_win)  # 返回 numpy
    else:
        # 改为：先加载原始尺寸，再 pad 到安全尺寸
        spikes_full_raw = load_spikes_from_open_dat(dat_path, 400, 400, 250)
        H_raw, W_raw = spikes_full_raw.shape[1], spikes_full_raw.shape[2]  # 250, 400
        # 目标尺寸：确保能被 16 整除
        target_H = ((H_raw + 15) // 16) * 16  # 256
        target_W = ((W_raw + 15) // 16) * 16  # 400 已经是 16 的倍数，所以还是 400
        # 用零填充（右、下）
        pad_bottom = target_H - H_raw
        pad_right  = target_W - W_raw
        spikes_full = np.pad(spikes_full_raw, ((0,0), (0,pad_bottom), (0,pad_right)), mode='constant', constant_values=0)
        print("计算 DSFT（GPU）...")
        # 后续 DSFT 初始化参数要改成 pad 后的尺寸
        dsft_calc = DSFT(spike_h=target_H, spike_w=target_W, device=device)
        dsft_full = dsft_calc.spikes2images(spikes_full, max_search_half_window=max_half_win)  # 返回 numpy


    # 对齐时间轴（全部保持在 CPU numpy 数组）
    start = max_half_win
    end = spikes_full.shape[0] - max_half_win
    spikes_aligned = spikes_full[start:end]   # numpy array
    dsft_aligned = dsft_full
    T_valid = spikes_aligned.shape[0]
    print(f"有效帧总数: {T_valid}")

    # 释放不再需要的全量 spikes_full 和 dsft_full（它们已切片，可删掉节省内存）
    del spikes_full, dsft_full
    torch.cuda.empty_cache()   # 清理 GPU 上可能残留的缓存
        
        
    # NR-IQA 评估配置  <--- 新增
    eval_freq = 1 # 每eval_freq个重建帧计算一次 IQA 分数
    save_iqa_path = 'iqa_scores.npz'
    # ================= 设备与模型 =================
    model = BSF().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    if any(k.startswith('module.') for k in checkpoint.keys()):
        new_state = OrderedDict()
        for k, v in checkpoint.items():
            new_state[k[7:]] = v
        model.load_state_dict(new_state)
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ================= 滑动窗口重建视频 =================
    half_win = window_size // 2
    center_start = half_win
    center_end = T_valid - half_win

    frame_list = []
    iqa_scores = {'BRISQUE': [],
                'PIQE': [], 
                'NIQE': []
                }
    lap_bsf = []
    mad_list = []
    kurt_list = []
    ratio_list = []
    for idx, center in enumerate(range(center_start, center_end, step)):
        t_start = center - half_win
        t_end   = center + half_win + 1

        spike_block = spikes_aligned[t_start:t_end]   # numpy (61,1000,988)
        dsft_block  = dsft_aligned[t_start:t_end]

        # 在循环内部，获得 center 之后
        # Gamma 校正：归一化 -> gamma -> 转回 uint8
        
        # 移动到 GPU
        spike_t = torch.from_numpy(spike_block).float().unsqueeze(0).to(device)
        dsft_t  = torch.from_numpy(dsft_block).float().unsqueeze(0).to(device)

        dsft_dict = convert_dsft4(spike=spike_t, dsft=dsft_t)
        input_dict = {'spikes': spike_t, 'dsft_dict': dsft_dict}

        with torch.no_grad():
            rec = model(input_dict=input_dict)

        img = rec.squeeze().cpu().numpy()
        print(f"rec.min: {rec.min()}, rec.max: {rec.max()}")
        img = np.clip(img, 0, 1)
        img_gamma = img ** (1/2.2) * 255.0
        img_gamma = img_gamma.astype(np.uint8)
        # img_gamma = np.log(1 + img * 255) / np.log(256) * 255
        # img_gamma = np.clip(img_gamma, 0, 255).astype(np.uint8)
        if not our_data:
            img_gamma = img_gamma[:250, :400]  # 裁剪回原始尺寸，去掉 pad 区域
        frame_list.append(img_gamma)
    
        lap_bsf.append(cv2.Laplacian(img_gamma, cv2.CV_64F).var())
        print(f"Laplacian var BSF: {lap_bsf[-1]:.2f}")  # 期望几百以上
        img_median = cv2.medianBlur(img_gamma, 3)
        residual = img_gamma.astype(np.float32) - img_median.astype(np.float32)
        mad = np.mean(np.abs(residual))
        kurt = kurtosis(residual.flatten(), fisher=False)
        ratio = nr_iqa.high_freq_energy_ratio(img_gamma)
        mad_list.append(mad)
        kurt_list.append(kurt)
        ratio_list.append(ratio)

        if idx == 0 or idx == 10:
            cv2.imwrite(f'{output_path}/frame_{idx:04d}.png', img_gamma)

        # ---------- NR-IQA 评估 ----------   <--- 新增块
        if idx % eval_freq == 0:
            scores = nr_iqa.compute_all(img_gamma)
            for metric_name, score_val in scores.items():
                iqa_scores[metric_name].append(score_val)
            print(f"Frame {idx:4d}: BRISQUE={scores['BRISQUE']:.3f}  PIQE={scores['PIQE']:.3f}  NIQE={scores['NIQE']:.3f}")
        # 显式释放当前窗口的 GPU 张量
        del spike_t, dsft_t, dsft_dict, input_dict, rec    
        torch.cuda.empty_cache()

        print(f"重建进度: {idx+1}/{ (center_end - center_start)//step + 1}")

    # ================= 保存视频 =================
    print(f"保存视频到 {output_video} ...")
    h, w = frame_list[0].shape
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')          # OpenCV 4.x 新 API
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h), isColor=False)
    for frame in frame_list:
        out.write(frame)
    out.release()

    # ================= 输出 NR-IQA 统计结果 =================   <--- 新增
    print("\n========== NR-IQA 评估结果 ==========")
    for metric, values in iqa_scores.items():
        if values:
            arr = np.array(values)
            print(f"{metric:8s}: mean={arr.mean():.4f} ± {arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
        else:
            print(f"{metric:8s}: 无数据")
    print(f"平均 Laplacian 变异数: {np.mean(lap_bsf):.2f}")
    print("\n========== 中值残差指标 ==========")
    print(f"残差 MAD  (均值±标准差): {np.mean(mad_list):.4f} ± {np.std(mad_list):.4f}")
    print(f"残差峰度 (均值±标准差): {np.mean(kurt_list):.4f} ± {np.std(kurt_list):.4f}")
    print(f"频域高频能量比 (均值±标准差): {np.mean(ratio_list):.4f} ± {np.std(ratio_list):.4f}")
    np.save(f"{output_path}/mad_bsf.npy", np.array(mad_list))
    np.save(f"{output_path}/kurtosis_bsf.npy", np.array(kurt_list)) 
    np.save(f"{output_path}/laplacian_var_bsf.npy", np.array(lap_bsf))
    np.save(f"{output_path}/high_freq_energy_ratio_bsf.npy", np.array(ratio_list))
    if save_iqa_path and any(iqa_scores.values()):
        base = os.path.splitext(save_iqa_path)[0]
        for metric, values in iqa_scores.items():
            if values:
                np.save(f"{output_path}/{base}_{metric}.npy", np.array(values))
        print(f"指标详细数据已保存到 {output_path}/{base}_BRISQUE.npy, {output_path}/{base}_PIQE.npy, {output_path}/{base}_NIQE.npy")

    print("完成！")