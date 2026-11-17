import numpy as np
from spikecv.spkData.load_dat import SpikeStream

dat_path = '/home/lhs183019/snap/snapd-desktop-integration/361/Desktop/BSF/raw_data_our/0.dat'  # 替换为你的 .dat 文件路径

vidar = SpikeStream(offline=True, camera_type=None,
                    filepath=dat_path, spike_w=1024, spike_h=1000,
                    print_dat_detail=False)
spikes_raw = vidar.get_file_matrix(flipud=False)  # (T, 1000, 1024)

# 看一下第 200 帧最右侧 30 列
import matplotlib.pyplot as plt
block = spikes_raw[0:50, :, -30:]   # 取前50帧，仍为 (50, 1000, 30)
mean_block = np.mean(block, axis=0) # (1000, 30)
plt.imshow(mean_block, cmap='hot', aspect='auto')
plt.title('Mean of frames 0-49, right 30 cols')
plt.colorbar()
plt.savefig('mean_right_edge.png', dpi=150)