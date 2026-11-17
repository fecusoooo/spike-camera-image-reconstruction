import numpy as np

def get_all_iqa_scores(prefix):
    iqa_scores_dict = {}
    for suffix in ['BRISQUE', 'PIQE', 'NIQE']:
        path = prefix + suffix + '.npy'
        scores = np.load(path, allow_pickle=True)
        print(f"{suffix:8s}: mean={scores.mean():.4f} ± {scores.std():.4f}")

def get_all_laplacian_vars(path):
    lap_vars = np.load(path, allow_pickle=True)
    print(f"平均 Laplacian 变异数: {lap_vars.mean():.2f} ± {lap_vars.std():.2f}")

def get_residual_metrics(mad_path, kurt_path):
    mad = np.load(mad_path, allow_pickle=True)
    kurt = np.load(kurt_path, allow_pickle=True)
    print(f"残差 MAD  (均值±标准差): {mad.mean():.4f} ± {mad.std():.4f}")
    print(f"残差峰度 (均值±标准差): {kurt.mean():.4f} ± {kurt.std():.4f}")

def get_high_freq_ratio(path):
    ratio = np.load(path, allow_pickle=True)
    print(f"频域高频能量比 (均值±标准差): {ratio.mean():.4f} ± {ratio.std():.4f}")

# ================= 参数 =================
print("TFI NR-IQA 评估结果")
print("======================")
prefix = '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/tfi/iqa_scores_'
get_all_iqa_scores(prefix)
get_all_laplacian_vars('/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/tfi/laplacian_var_tfi.npy')
get_residual_metrics('/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/tfi/mad_tfi.npy',
                    '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/tfi/kurtosis_tfi.npy')
get_high_freq_ratio('/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/tfi/high_freq_energy_ratio_tfi.npy')


print("\nTFP NR-IQA 评估结果")
print("======================")
prefix = '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/tfp/iqa_tfp_'
get_all_iqa_scores(prefix)
get_all_laplacian_vars('/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/tfp/laplacian_var_tfp.npy')
get_residual_metrics('/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/tfp/mad_tfp.npy',
                    '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/tfp/kurtosis_tfp.npy')
get_high_freq_ratio('/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/tfp/high_freq_energy_ratio_tfp.npy')

print("\nSSML NR-IQA 评估结果")
print("======================")
prefix = '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/ssml/iqa_scores_'
get_all_iqa_scores(prefix)
get_all_laplacian_vars('/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/ssml/laplacian_var_ssml.npy')


print("\nBSF NR-IQA 评估结果")
print("======================")
prefix = '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/bsf/iqa_scores_'
get_all_iqa_scores(prefix)
get_all_laplacian_vars('/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/bsf/laplacian_var_bsf.npy')
get_residual_metrics('/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/bsf/mad_bsf.npy',
                    '/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/bsf/kurtosis_bsf.npy')
get_high_freq_ratio('/home/lhs183019/snap/snapd-desktop-integration/current/Desktop/BSF/my_result/bsf/high_freq_energy_ratio_bsf.npy')