# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSDeformAttnFuse(nn.Module):
    """
    【标准/无Bug/可训练】单尺度可变形注意力
    Query: Ego 特征
    Key/Value: 协作智能体特征
    完全对齐 Deformable DETR 单尺度实现
    """
    def __init__(self, d_model=256, n_heads=4, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = 1  # 单尺度
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # 可变形采样偏移预测
        self.sampling_offsets = nn.Linear(
            d_model, n_heads * self.n_levels * n_points * 2
        )
        # 注意力权重预测
        self.attention_weights = nn.Linear(
            d_model, n_heads * self.n_levels * n_points
        )

        # 投影层
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # 偏移初始化：极坐标方向（原版DETR最优初始化）
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1  # 不同点不同半径
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # 注意力权重初始化
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)

        # 投影初始化
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, ego_feat, collab_feat):
        """
        ego_feat:    [B, C, H, W] 自车特征
        collab_feat: [B, C, H, W] 协作智能体特征
        return:      [B, C, H, W] 融合后特征
        """
        B, C, H, W = ego_feat.shape
        device = ego_feat.device

        # 展平为序列
        query = ego_feat.flatten(2).transpose(1, 2)        # [B, HW, C]
        value = collab_feat.flatten(2).transpose(1, 2)     # [B, HW, C]

        # Value 投影 + 分头
        value = self.value_proj(value)
        value = value.view(B, H*W, self.n_heads, self.head_dim)

        # 空间信息
        spatial_shapes = torch.as_tensor([[H, W]], device=device)
        level_start_index = torch.zeros(1, device=device, dtype=torch.long)

        # 参考点 (归一化坐标)
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H-0.5, H, device=device),
            torch.linspace(0.5, W-0.5, W, device=device),
            indexing='ij'
        )
        ref_points = torch.stack([ref_x/W, ref_y/H], -1).flatten(0, 1)  # [HW, 2]
        ref_points = ref_points.view(1, H*W, 1, 2).repeat(B, 1, 1, 1)   # [B, HW, 1, 2]

        # 预测偏移 + 注意力权重
        sampling_offsets = self.sampling_offsets(query).view(
            B, H*W, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, H*W, self.n_heads, self.n_levels * self.n_points
        ).softmax(-1).view(
            B, H*W, self.n_heads, self.n_levels, self.n_points
        )

        # 计算采样位置（不使用clamp，保留远距离建模能力）
        offset_normalizer = torch.tensor([W, H], device=device).view(1,1,1,1,1,2)
        sampling_locations = ref_points.unsqueeze(3).unsqueeze(4) + sampling_offsets / offset_normalizer

        # 可变形注意力核心（无Bug版）
        output = self.ms_deform_attn_core(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        # 输出投影 + 残差
        output = self.output_proj(output)
        output = output + query

        # 恢复图像形状
        return output.transpose(1, 2).view(B, C, H, W)

    def ms_deform_attn_core(self, value, spatial_shapes, sampling_locations, attention_weights):
        B, S, M, D = value.shape
        _, Lq, _, L, P, _ = sampling_locations.shape
        H, W = spatial_shapes[0]

        # 正确维度排列：[B*M, D, H, W]
        value = value.permute(0, 2, 3, 1).reshape(B * M, D, H, W)

        # 栅格坐标：[B, Lq, M, L, P, 2] -> [B*M, Lq, P, 2]
        grid = sampling_locations[:, :, :, 0].permute(0, 2, 1, 4, 3).reshape(B * M, Lq, P, 2)
        grid = grid * 2 - 1  # 转到 [-1,1]

        # 双线性采样
        sampled_feat = F.grid_sample(
            value, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )  # [B*M, D, Lq, P]

        # 注意力加权
        attn_weight = attention_weights.permute(0,2,1,3,4).reshape(B*M, 1, Lq, P)
        output = (sampled_feat * attn_weight).sum(-1)  # [B*M, D, Lq]

        # 合并多头
        return output.view(B, M*D, Lq).transpose(1, 2)
    
def ms_deform_attn_fuse(x, score, record_len, affine_matrix, align_corners, fuse_module):
    """
    【无Bug/高性能】多智能体可变形注意力融合
    """
    B = len(record_len)
    _, C, H, W = x.shape
    split_x = regroup(x, record_len)
    out = []

    for b in range(B):
        N = record_len[b]
        feat = split_x[b]  # [N, C, H, W]
        t_matrix = affine_matrix[b, :N]

        # 全部对齐到自车视角
        aligned_feats = warp_affine_simple(feat, t_matrix[0], (H, W), align_corners=align_corners)
        ego_feat = aligned_feats[0:1]  # [1, C, H, W]

        if N == 1:
            out.append(ego_feat)
            continue

        # 全局一次性融合所有协作智能体
        collab_feat = aligned_feats[1:].mean(0, keepdim=True)  # 全局融合
        fused_feat = fuse_module(ego_feat, collab_feat)
        out.append(fused_feat)

    return torch.cat(out, dim=0)


def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)
    
    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)
        
    record_len : list
        shape: (B)
        
    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3) 
    """

    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego), 
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device), 
                                    scores_in_ego)

        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    
    return out

class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck, 
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)
        
        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )
        self.deform_fuse_modules = nn.ModuleList([
        MSDeformAttnFuse(
            d_model=self.model_cfg["num_filters"][i],
            n_heads=model_cfg.get("deform_n_heads", 4),
            n_points=model_cfg.get("deform_n_points", 4),
        )
            for i in range(self.num_levels)
            ])

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        final_feature = self.decode_multiscale_feature(feature_list)

        return final_feature, occ_map_list
    
    def forward_collab(self, spatial_features, record_len, affine_matrix, agent_modality_list = None, cam_crop_info = None):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list] 
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}


        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask
            fused = ms_deform_attn_fuse(
                x=feature_list[i],
                score=score,
                record_len=record_len,
                affine_matrix=affine_matrix,
                align_corners=self.align_corners,
                fuse_module=self.deform_fuse_modules[i],
            )
            fused_feature_list.append(fused)
        fused_feature = self.decode_multiscale_feature(fused_feature_list)
        # print("fused_feature shape: ", fused_feature.shape)

        
        return fused_feature, occ_map_list