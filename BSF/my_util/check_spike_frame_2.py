import numpy as np
import matplotlib.pyplot as plt

dat_path = '/home/lhs183019/snap/snapd-desktop-integration/361/Desktop/BSF/raw_data_our/0.dat'
raw_data = np.fromfile(dat_path, dtype=np.uint8)
print(f"文件总字节: {len(raw_data)}")

# ====== 1. 推断可能的帧尺寸 ======
# 你的相机: 1000×1000 像素 + 24 测试列 = 1000×1024
# 文件大小应该是 height * width_bits//8 * n_frames
total_bytes = len(raw_data)
for H in [1000]:
    for W_bits in [1024, 1000, 1032, 1040, 1056]:
        fb = H * W_bits // 8
        if total_bytes % fb == 0:
            nf = total_bytes // fb
            print(f"可能尺寸: H={H}, W_bits={W_bits}, frame_bytes={fb}, 总帧数={nf}")

# ====== 2. 取出第一帧，按行主序和列主序各解码一次 ======
# 假设最可能的尺寸
H, W_decode = 1000, 1024
frame_bytes = H * W_decode // 8
n_frames = total_bytes // frame_bytes
data = raw_data[:n_frames * frame_bytes]

# --- 2a. 行主序（Row-major）---
data_row = data.reshape(n_frames, H, W_decode // 8)
bits_row = np.unpackbits(data_row[0], axis=-1, bitorder='little')
# bits_row.shape = (1000, 1024)

# --- 2b. 列主序（Column-major）---
data_col = data.reshape(n_frames, W_decode // 8, H).transpose(0, 2, 1)
# → (n_frames, H, W_decode//8)
bits_col = np.unpackbits(data_col[0], axis=-1, bitorder='little')

# ====== 3. 可视化比较 ======
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 行主序的重建（删除测试列后显示）
delete_test = np.arange(1000, 1024)
img_row = np.delete(bits_row, delete_test, axis=1)
axes[0].imshow(img_row, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Row-major (当前假设)', fontsize=14)
axes[0].axis('off')

# 列主序的重建（同样删除）
img_col = np.delete(bits_col, delete_test, axis=1)
axes[1].imshow(img_col, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('Column-major (可能正确)', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('diagnostic_row_vs_col.png', dpi=150)
plt.show()

# ====== 4. 检查测试列位置 ======
print("\n===== 行主序下各列均值 =====")
for c in range(1024):
    m = bits_row[:, c].mean()
    if m < 0.05 or m > 0.95:
        print(f"  列 {c:4d}: mean={m:.4f}  ← 异常（全{'0' if m < 0.05 else '1'}）")

print("\n===== 列主序下各列均值 =====")
for c in range(1024):
    m = bits_col[:, c].mean()
    if m < 0.05 or m > 0.95:
        print(f"  列 {c:4d}: mean={m:.4f}  ← 异常（全{'0' if m < 0.05 else '1'}）")