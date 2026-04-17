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
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math

class MSDeformAttnFuse(nn.Module):
    """
    标准 MSDeformAttn 融合（单尺度版本）
    Query: ego
    Key/Value:  agent
    """

    def __init__(self, d_model, n_heads=4, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = 1   # 单尺度
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * self.n_levels * n_points * 2
        )
        self.attention_weights = nn.Linear(
            d_model, n_heads * self.n_levels * n_points
        )

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):


        # ===================== 【升级点】原版极坐标初始化 =====================
        constant_(self.sampling_offsets.weight.data, 0.)
        # 极坐标方向初始化 (和 MSDeformAttn 完全一致)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # 归一化 + 维度适配 [n_heads, n_levels, n_points, 2]
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])\
            .view(self.n_heads, 1, 1, 2)\
            .repeat(1, self.n_levels, self.n_points, 1)
        # 不同采样点不同半径
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
            # grid_init[:, :, i, :] *= (i + 1) / self.n_points
        # 赋值偏置
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # ====================================================================
        # constant_(self.sampling_offsets.weight, 0.)
        constant_(self.attention_weights.weight, 0.)
        constant_(self.attention_weights.bias, 0.)

        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias, 0.)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias, 0.)

    def forward(self, ego_feat, collab_feat):
        """
        ego_feat:    [1, C, H, W]
        collab_feat: [1, C, H, W]
        """

        _, C, H, W = collab_feat.shape
        device = ego_feat.device

        # ---------------------------
        # flatten
        # ---------------------------
        query = ego_feat.flatten(2).transpose(1, 2)       # [1, HW, C]
        value = collab_feat.flatten(2).transpose(1, 2)    # [1, HW, C]

        # 拼接所有 agent
        value = value.reshape(1,  H * W, C)

        value = self.value_proj(value)
        value = value.view(1,  H * W, self.n_heads, C // self.n_heads)

        # ---------------------------
        # spatial shapes
        # ---------------------------
        spatial_shapes = torch.as_tensor(
            [[H, W]], dtype=torch.long, device=device
        )

        level_start_index = torch.as_tensor(
            [0], dtype=torch.long, device=device
        )

        # ---------------------------
        # reference points
        # ---------------------------
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device),
            torch.linspace(0.5, W - 0.5, W, device=device),
            indexing='ij'
        )

        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W

        ref_points = torch.stack((ref_x, ref_y), -1)  # [HW, 2]
        ref_points = ref_points.unsqueeze(0).unsqueeze(2)  # [1, HW, 1, 2]

        # ---------------------------
        # offsets & weights
        # ---------------------------
        sampling_offsets = self.sampling_offsets(query).view(
            1, H * W, self.n_heads, self.n_levels, self.n_points, 2
        )

        attention_weights = self.attention_weights(query).view(
            1, H * W, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            1, H * W, self.n_heads, self.n_levels, self.n_points
        )

        # ---------------------------
        # sampling locations
        # ---------------------------
        offset_normalizer = torch.tensor(
            [W, H], dtype=torch.float32, device=device
        )

        sampling_locations = (
            ref_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer
        )

    #     sampling_locations = torch.clamp(
    #     ref_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer,
    #     min=0.0, max=1.0
    # )

        # ---------------------------
        # deformable attention core
        # ---------------------------
        output = self.ms_deform_attn_core_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)

        # residual
        output = output + query

        return output.transpose(1, 2).reshape(1, C, H, W)

    @staticmethod
    def ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights):
        """
        直接复用 mrcnet 的实现（单尺度版）
        """
        N_, S_, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape

        H_, W_ = spatial_shapes[0]

        value_l_ = value.reshape(N_, H_ * W_, M_, D_).flatten(2).transpose(1, 2)
        value_l_ = value_l_.reshape(N_ * M_, D_, H_, W_)

        sampling_grid = 2 * sampling_locations - 1
        sampling_grid = sampling_grid[:, :, :, 0].transpose(1, 2).flatten(0, 1)

        sampled = F.grid_sample(
            value_l_,
            sampling_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        sampled = sampled.view(N_ * M_, D_, Lq_, P_)
        attention_weights = attention_weights.permute(0, 2, 3, 1, 4).reshape(N_ * M_, 1, Lq_, P_)
        output = (sampled * attention_weights).sum(-1)
        output = output.view(N_, M_ * D_, Lq_)
        return output.transpose(1, 2)


def ms_deform_attn_fuse(x,score, record_len, affine_matrix,
                       align_corners, fuse_module):
    """
    x: (sum(N), C, H, W)
    score: (sum(N), 1, H, W)
    """

    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]

    split_x = regroup(x, record_len)
    split_score = regroup(score, record_len)

    out = []

    for b in range(B):
        N = record_len[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]

        i = 0  # ego

        # warp到ego
        feature_in_ego = warp_affine_simple(
            split_x[b], t_matrix[i], (H, W),
            align_corners=align_corners
        )  # [N, C, H, W]

        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego), 
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device), 
                                    scores_in_ego)
        feature_in_ego = feature_in_ego * scores_in_ego

        ego_feat = feature_in_ego[0:1]

        if N == 1:
            out.append(ego_feat.squeeze(0))
            continue

        fused = ego_feat.clone()
        for n in range(1, N):
            collab_feat = feature_in_ego[n:n+1]          # [1, C, H, W]
            attended = fuse_module(ego_feat, collab_feat)
            fused = fused + attended

        fused = fused/N
        out.append(fused.squeeze(0))

    return torch.stack(out)


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