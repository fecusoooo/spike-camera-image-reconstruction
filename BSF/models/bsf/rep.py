import torch
import torch.nn as nn


class MODF(nn.Module):
    """
    PG-MODF: Physics-Guided Multi-order DSFT Fusion

    原始 BSF:
        R = feat_11 + other_feat_res

    改进后:
        R = feat_11 + physics_gate * other_feat_res

    作用:
        利用低阶与高阶 DSFT 差异构造门控，无需训练即可运行。
        高阶 DSFT 有助于抑制 spike fluctuation，但在高速运动/边缘区域可能引入过平滑。
        physics_gate 根据局部一致性自动控制高阶 DSFT 注入强度。
    """

    def __init__(self, base_dim=64, act=nn.ReLU()):
        super().__init__()
        self.base_dim = base_dim

        # 原始 MODF 的共享特征提取器
        self.conv1 = self._make_layer(
            input_dim=21,
            hidden_dim=self.base_dim,
            output_dim=self.base_dim,
            act=act
        )

        # 处理三种高阶 DSFT: (1,2), (2,1), (2,2)
        self.conv_for_others = nn.ModuleList([
            self._make_layer(
                input_dim=self.base_dim,
                hidden_dim=self.base_dim,
                output_dim=self.base_dim,
                act=act
            )
            for _ in range(3)
        ])

        # 融合高阶 DSFT 特征
        self.conv_fuse = self._make_layer(
            input_dim=self.base_dim * 3,
            hidden_dim=self.base_dim,
            output_dim=self.base_dim,
            act=act
        )

        # 注意: 不再使用可训练 gate
        # self.gate 已经被移除

    def _make_layer(self, input_dim, hidden_dim, output_dim, act):
        layer = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            act,
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1),
        )
        return layer

    def forward(self, dsft_dict):
        # 保持原论文的 reciprocal DSFT 形式
        d11 = 1.0 / dsft_dict['dsft11']
        d12 = 2.0 / dsft_dict['dsft12']
        d21 = 2.0 / dsft_dict['dsft21']
        d22 = 3.0 / dsft_dict['dsft22']

        d_list = [d11, d12, d21, d22]

        # 共享特征提取
        feat_batch_cat = self.conv1(torch.cat(d_list, dim=0))
        feat_list = feat_batch_cat.chunk(4, dim=0)

        feat_11 = feat_list[0]
        feat_others_list = feat_list[1:]

        feat_others_list_processed = []
        for ii in range(3):
            feat_others_list_processed.append(
                self.conv_for_others[ii](feat_others_list[ii])
            )

        other_feat = torch.cat(feat_others_list_processed, dim=1)
        other_feat_res = self.conv_fuse(other_feat)

        # ========================
        # 物理先验门控 PG-MODF
        # ========================
        eps = 1e-6
        high_mean = (d12 + d21 + d22) / 3.0
        diff = torch.abs(high_mean - d11) / (torch.abs(d11) + eps)
        high_stack = torch.stack([d12, d21, d22], dim=0)
        high_std = torch.std(high_stack, dim=0)
        high_var = high_std / (torch.abs(high_mean) + eps)
        diff_map = torch.mean(diff, dim=1, keepdim=True)
        var_map = torch.mean(high_var, dim=1, keepdim=True)

        beta = 0.2  # 控制门控强度
        physics_gate = torch.exp(-beta * (diff_map + 0.5 * var_map))
        physics_gate = torch.clamp(physics_gate, 0.0, 1.0)
        physics_gate = physics_gate.expand(-1, feat_11.shape[1], -1, -1)

        # 融合
        return feat_11 + physics_gate * other_feat_res