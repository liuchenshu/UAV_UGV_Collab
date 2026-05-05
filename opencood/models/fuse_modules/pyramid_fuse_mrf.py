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
from opencood.models.pose_error_correction.mrcnet_fusion import MSRobustFusion

def weighted_fuse_backup(x, score, record_len, affine_matrix, align_corners):
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
        # feature_in_ego = feature_in_ego * scores_in_ego
        out.append(feature_in_ego)
        # out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    
    return out

def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    Returns:
        out: (B, N_max, C, H, W)  ← 改为固定形状，不足的 agent 用 0 填充
    """
    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    N_max = max(record_len)                      # ← 新增：确定最大 agent 数

    split_x     = regroup(x,     record_len)
    split_score = regroup(score, record_len)

    out = []
    for b in range(B):
        N       = record_len[b]
        score_b = split_score[b]                 # ← 改：避免覆盖外层 score 变量
        t_matrix = affine_matrix[b][:N, :N, :, :]

        feature_in_ego = warp_affine_simple(
            split_x[b], t_matrix[0, :, :, :], (H, W), align_corners=align_corners)
        scores_in_ego  = warp_affine_simple(
            score_b,     t_matrix[0, :, :, :], (H, W), align_corners=align_corners)

        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(
            torch.isnan(scores_in_ego),
            torch.zeros_like(scores_in_ego),
            scores_in_ego)

        feature_in_ego = feature_in_ego * scores_in_ego   # (N, C, H, W)

        # ── zero-pad 到 N_max ──────────────────────────────────────────────
        if N < N_max:
            pad = torch.zeros(N_max - N, C, H, W, device=feature_in_ego.device)
            feature_in_ego = torch.cat([feature_in_ego, pad], dim=0)

        out.append(feature_in_ego)                         # (N_max, C, H, W)

    return torch.stack(out)                                # (B, N_max, C, H, W)

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
        num_agents = model_cfg['mrf_fusion']['num_agents']
        self.mrf_fusion = MSRobustFusion(num_agents, model_cfg['mrf_fusion'])
        

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
        MS_feature_list = []
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
            # fused_feature_list : fused_feature_list[0] shape:  torch.Size([1, 64, 256, 256])
                                # fused_feature_list[1] shape:  torch.Size([1, 128, 128, 128])
                                # fused_feature_list[2] shape:  torch.Size([1, 256, 64, 64])
            # fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
            MS_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
            # stack fused_feature_list[0] shape:  torch.Size([1, 3, 64, 256, 256])
                    # fused_feature_list[1] shape:  torch.Size([1, 3, 128, 128, 128])
                    # fused_feature_list[2] shape:  torch.Size([1, 3, 256, 64, 64])
            
        # B = len(record_len)
        # batch_dict_list = []

        # for b in range(B):
        #         N = record_len[b]
        #         for n in range(N):
        #             batch_dict = {}
        #             for level_idx in range(self.num_levels):
        #                 scale = 2 ** (level_idx + 1)
        #                 feature_key = f'src_features_for_align_{scale}x'
        #                 # 从 (B, N, C, H, W) 提取 (1, C, H, W) 并添加回batch维度
        #                 batch_dict[feature_key] = MS_feature_list[level_idx][b:b+1, n, :, :, :]
        #             batch_dict_list.append(batch_dict)
            
        #     # ============ 调用 MSRobustFusion 进行多尺度融合 ============
        # fused_output, comm_volume ,fused_output_encode= self.mrf_fusion(batch_dict_list, record_len)

        N_max = max(record_len)
        batch_dict_list = []

        for n in range(N_max):                          # 按 agent 索引，共 N_max 个 dict
            agent_dict = {}
            for level_idx in range(self.num_levels):
                scale = 2 ** (level_idx + 1)
                # MS_feature_list[level_idx]: (B, N_max, C, H, W)
                # 切出第 n 个 agent 在所有 batch 中的特征 → (B, C, H, W)
                agent_dict[f'src_features_for_align_{scale}x'] = \
                    MS_feature_list[level_idx][:, n, :, :, :]           # (B, C, H, W) ✓
            batch_dict_list.append(agent_dict)
        fused_output, comm_volume, fused_output_encode = self.mrf_fusion(batch_dict_list, record_len)
            
            # ============ 提取多尺度融合特征进行最终解码 ============
        # fused_feature_list = []
        # for level_idx in range(self.num_levels):
        #         scale = 2 ** (level_idx + 1)
        #         # 从 MSRobustFusion 输出中直接获取对应 scale 的融合特征
        #         feature_key = f'spatialAttn_result_{scale}x'
        #         if feature_key in fused_output:
        #             # fused_output[feature_key] 的形状是 (B, H*W, C)，需要转换为 (B, C, H, W)
        #             feat_flat = fused_output[feature_key]  # (B, H*W, C)
        #             H, W = MS_feature_list[level_idx].shape[3:]
        #             feat_2d = feat_flat.transpose(1, 2).view(-1, feat_flat.shape[-1], H, W)  # (B, C, H, W)
        #             fused_feature_list.append(feat_2d)

        # ============ 提取多尺度融合特征进行最终解码 ============
        # fused_feature_list = []
        # for level_idx in range(self.num_levels):
        #     scale = 2 ** (level_idx + 1)
        #     # 从 MSRobustFusion 输出中直接获取对应 scale 的融合特征
        #     feature_key = f'spatialAttn_result_{scale}x'
            
        #     # fused_output[feature_key] 的形状是 (B, H*W, C)，需要转换为 (B, C, H, W)
        #     feat_flat = fused_output[feature_key]  # (B, H*W, C)
        #     H, W = MS_feature_list[level_idx].shape[3:]
        #     feat_2d = feat_flat.transpose(1, 2).view(-1, feat_flat.shape[-1], H, W)  # (B, C, H, W)
        #     fused_feature_list.append(feat_2d)
            
        # fused_feature = self.decode_multiscale_feature(fused_feature_list)
        fused_feature = fused_output['aggregated_spatial_features_2d']
        # print("fused_feature shape: ", fused_feature.shape)

        
        return fused_feature, occ_map_list 