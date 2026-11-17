"""
nr_iqa.py
无参考图像质量评估工具 (使用 pyiqa 库)
支持 BRISQUE, PIQE, NIQE 三个指标
"""

import torch
import numpy as np
import pyiqa
import warnings

# 抑制警告信息
warnings.filterwarnings('ignore', category=FutureWarning)

def get_device():
    """自动选择设备: GPU 或 CPU"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def high_freq_energy_ratio(img, threshold=0.4):
    """计算图像频域中归一化频率 > threshold 的能量占比，用于检测高频规则噪点"""
    f = np.fft.fft2(img.astype(np.float32) / 255.0)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    total_energy = np.sum(magnitude ** 2)
    h, w = magnitude.shape
    Y, X = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    norm_dist = dist / np.sqrt(cy**2 + cx**2)
    mask = norm_dist > threshold
    high_energy = np.sum(magnitude[mask] ** 2)
    return high_energy / total_energy if total_energy > 0 else 0.0

def compute_brisque(img: np.ndarray) -> float:
    """计算 BRISQUE 分数 (值越低越好)"""
    # 1. 创建评估器 (首次调用会自动下载模型)
    brisque_metric = pyiqa.create_metric('brisque', device=get_device())
    
    # 2. 转换图像: uint8 (H,W) -> tensor (1,1,H,W), 并归一化到 [0,1]
    img_tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0).to(get_device())
    
    # 3. 计算分数并返回
    score = brisque_metric(img_tensor)
    return float(score.cpu().numpy())

def compute_piqe(img: np.ndarray) -> float:
    """计算 PIQE 分数 (值越低越好)"""
    piqe_metric = pyiqa.create_metric('piqe', device=get_device())
    # 确保形状为 (1, 1, H, W)
    img_tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0)
    # 复制为3通道 -> (1, 3, H, W)
    img_tensor = img_tensor.repeat(1, 3, 1, 1)
    img_tensor = img_tensor.to(get_device())
    score = piqe_metric(img_tensor)
    return float(score.cpu().numpy())

def compute_niqe(img: np.ndarray) -> float:
    """计算 NIQE 分数 (值越低越好)"""
    niqe_metric = pyiqa.create_metric('niqe', device=get_device())
    # 确保形状为 (1, 1, H, W)
    img_tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0)
    # 复制为3通道 -> (1, 3, H, W)
    img_tensor = img_tensor.repeat(1, 3, 1, 1)
    img_tensor = img_tensor.to(get_device())
    try:
        score = niqe_metric(img_tensor)
        return float(score.cpu().numpy())
    except Exception as e:
        print(f"NIQE 计算失败: {e}")
        return 0.0

def compute_all(img: np.ndarray) -> dict:
    """一次性计算所有三个指标，返回字典"""
    scores = {}
    try:
        scores['BRISQUE'] = compute_brisque(img)
    except Exception as e:
        print(f"BRISQUE 计算失败: {e}")
        scores['BRISQUE'] = 0.0
    
    try:
        scores['PIQE'] = compute_piqe(img)
    except Exception as e:
        print(f"PIQE 计算失败: {e}")
        scores['PIQE'] = 0.0
    
    try:
        scores['NIQE'] = compute_niqe(img)
    except Exception as e:
        print(f"NIQE 计算失败: {e}")
        scores['NIQE'] = 0.0
    
    return scores

if __name__ == "__main__":
    import sys
    import cv2
    
    if len(sys.argv) != 2:
        print("Usage: python nr_iqa.py <image_path>")
        sys.exit(1)
    
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image")
        sys.exit(1)
    
    scores = compute_all(img)
    print(" | ".join([f"{k}: {v:.4f}" for k, v in scores.items()]))

