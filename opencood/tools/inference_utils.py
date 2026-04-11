# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils.transformation_utils import get_relative_transformation
from opencood.utils.box_utils import create_bbx, project_box3d, nms_rotated
from opencood.utils.camera_utils import indices_to_depth
from sklearn.metrics import mean_squared_error

def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict



def inference_no_fusion(batch_data, model, dataset, single_gt=False):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    single_gt : bool
        if True, only use ego agent's label.
        else, use all agent's merged labels.
    """
    output_dict_ego = OrderedDict()
    if single_gt:
        batch_data = {'ego': batch_data['ego']}
        
    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_no_fusion(batch_data,  # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict

def inference_no_fusion_w_uncertainty(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
        dataset.post_process_no_fusion_uncertainty(batch_data, # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor, \
                    "uncertainty_tensor" : uncertainty_tensor}

    return return_dict


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)
    # print("output_dict['ego'] keys: ", output_dict['ego'].keys())
    # print("output_dict['ego']['cls_preds_single'] shape: ", output_dict['ego']['cls_preds_single'].shape)
    # print("output_dict['ego']['cls_preds']: ", output_dict['ego']['cls_preds'].shape)
    # print("output_dict['ego']['reg_preds_single'] shape: ", output_dict['ego']['reg_preds_single'].shape)
    # print("output_dict['ego']['reg_preds'] shape: ", output_dict['ego']['reg_preds'].shape)
    # print("output_dict['ego']['dir_preds_single'] shape: ", output_dict['ego']['dir_preds_single'].shape)
    # print("output_dict['ego']['dir_preds'] shape: ", output_dict['ego']['dir_preds'].shape)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    return return_dict

def inference_intermediate_fusion_withsingle(batch_data, model, dataset):
    """
    Model inference for intermediate fusion with single cav.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.IntermediateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()
    output_dict_ego['ego'] = model(batch_data['ego'])
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data, output_dict_ego)
    print("pred_box_tensor shape: ", pred_box_tensor.shape)
    print("pred_score shape: ", pred_score.shape)
    print("gt_box_tensor shape: ", gt_box_tensor.shape)
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    agent_number= output_dict_ego['ego']['cls_preds_single'].shape[0]
    return_dict.update({"agent_number": agent_number})
    # agent_number = batch_data['ego']['record_len'][0].item() if 'record_len' in batch_data['ego'] else output_dict_ego['ego']['cls_preds_single'].shape[0]
    # return_dict.update({"agent_number": agent_number})
    print("batch_data['ego'].key: ", batch_data['ego'].keys())
    pairwise_t_matrix = batch_data['ego']['pairwise_t_matrix']
    print(f"Transform matrix: {pairwise_t_matrix}")
    for i in range(agent_number):
        output_dict_ego_single = OrderedDict({"ego": {}})
        output_dict_ego_single['ego']['cls_preds'] = output_dict_ego['ego']['cls_preds_single'][i:i+1]
        output_dict_ego_single['ego']['reg_preds'] = output_dict_ego['ego']['reg_preds_single'][i:i+1]
        output_dict_ego_single['ego']['dir_preds'] = output_dict_ego['ego']['dir_preds_single'][i:i+1]
        # print(f"output_dict_ego_single['ego']['cls_preds'] shape: ", output_dict_ego_single['ego']['cls_preds'].shape)
        pred_box_tensor_single, pred_score_single, gt_box_tensor_single = \
            dataset.post_process(batch_data, output_dict_ego_single)

        print(f"pred_box_tensor_single_{i} shape: ", pred_box_tensor_single.shape)
        print(f"pred_score_single_{i} shape: ", pred_score_single.shape)
        print(f"gt_box_tensor_single_{i} shape: ", gt_box_tensor_single.shape)
        if(i != 0):
            pred_box_tensor_single = project_box3d(pred_box_tensor_single, pairwise_t_matrix[0, i, 0])
        return_dict.update({f"pred_box_tensor_single_{i}": pred_box_tensor_single,
                            f"pred_score_single_{i}": pred_score_single,
                            f"gt_box_tensor_single_{i}": gt_box_tensor_single}) 
    return return_dict

def inference_intermediate_fusion_withsingle_v1(batch_data, model, dataset):
    """
    Intermediate fusion inference + per-agent single-branch decoding.
    Assumes inference batch_size == 1.
    """
    output_dict_ego = OrderedDict()
    output_dict_ego["ego"] = model(batch_data["ego"])

    # fused prediction (original collaborative branch)
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data, output_dict_ego)

    return_dict = {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor
    }

    ego_out = output_dict_ego["ego"]
    if "cls_preds_single" not in ego_out or "reg_preds_single" not in ego_out:
        # single head not available, return fused only
        return return_dict

    cls_single = ego_out["cls_preds_single"]   # [N_agent_total, C, H, W]
    reg_single = ego_out["reg_preds_single"]   # [N_agent_total, 7*anchor, H, W]
    dir_single = ego_out.get("dir_preds_single", None)

    # For inference, batch_size is 1 => record_len has one element
    record_len = batch_data["ego"]["record_len"]
    if torch.is_tensor(record_len):
        num_agents = int(record_len[0].item())
    else:
        num_agents = int(record_len[0])

    return_dict["agent_number"] = num_agents

    # pairwise_t_matrix shape: [B, L, L, 4, 4], pairwise[i, j] = T_{j<-i}
    # agent i -> ego(0) => pairwise[0, i, 0]
    pairwise_t = batch_data["ego"]["pairwise_t_matrix"][0]
    anchor_box = batch_data["ego"]["anchor_box"]

    for i in range(num_agents):
        # IMPORTANT: create fresh dict each loop (no shallow copy)
        single_data_dict = OrderedDict({
            "ego": {
                "anchor_box": anchor_box,
                "transformation_matrix": pairwise_t[i, 0].float(),
                "transformation_matrix_clean": pairwise_t[i, 0].float()
            }
        })

        single_out_ego = {
            "cls_preds": cls_single[i:i+1],
            "reg_preds": reg_single[i:i+1]
        }
        if dir_single is not None:
            single_out_ego["dir_preds"] = dir_single[i:i+1]

        output_dict_ego_single = OrderedDict({"ego": single_out_ego})

        pred_box_tensor_single, pred_score_single, gt_box_tensor_single = \
            dataset.post_processor.post_process(single_data_dict, output_dict_ego_single)

        return_dict.update({
            f"pred_box_tensor_single_{i}": pred_box_tensor_single,
            f"pred_score_single_{i}": pred_score_single,
            f"gt_box_tensor_single_{i}": gt_box_tensor_single
        })

    return return_dict

def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return_dict = inference_early_fusion(batch_data, model, dataset)
    return return_dict





def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)


def depth_metric(depth_items, grid_conf):
    # depth logdit: [N, D, H, W]
    # depth gt indices: [N, H, W]
    depth_logit, depth_gt_indices = depth_items
    depth_pred_indices = torch.argmax(depth_logit, 1)
    depth_pred = indices_to_depth(depth_pred_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    depth_gt = indices_to_depth(depth_gt_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    rmse = mean_squared_error(depth_gt.cpu(), depth_pred.cpu(), squared=False)
    return rmse


def fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data):
    """
    Fix the missing pred_box and gt_box for ego and cav(s).
    Args:
        pred_box_tensor : tensor
            shape (N1, 8, 3), may or may not include ego agent prediction, but it should include
        gt_box_tensor : tensor
            shape (N2, 8, 3), not include ego agent in camera cases, but it should include
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    Returns:
        pred_box_tensor : tensor
            shape (N1+?, 8, 3)
        gt_box_tensor : tensor
            shape (N2+1, 8, 3)
    """
    if pred_box_tensor is None or gt_box_tensor is None:
        return pred_box_tensor, gt_box_tensor, pred_score, 0
    # prepare cav's boxes

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()
        
    extent = [2.45, 1.06, 0.75]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_tensor = torch.tensor(np.concatenate(box_list, axis=0), device=pred_box_tensor.device)
    
    pred_box_tensor_ = torch.cat((cav_box_tensor, pred_box_tensor), dim=0)
    gt_box_tensor_ = torch.cat((cav_box_tensor, gt_box_tensor), dim=0)

    pred_score_ = torch.cat((torch.ones(N, device=pred_score.device), pred_score))

    gt_score_ = torch.ones(gt_box_tensor_.shape[0], device=pred_box_tensor.device)
    gt_score_[N:] = 0.5

    keep_index = nms_rotated(pred_box_tensor_,
                            pred_score_,
                            0.01)
    pred_box_tensor = pred_box_tensor_[keep_index]
    pred_score = pred_score_[keep_index]

    keep_index = nms_rotated(gt_box_tensor_,
                            gt_score_,
                            0.01)
    gt_box_tensor = gt_box_tensor_[keep_index]

    return pred_box_tensor, gt_box_tensor, pred_score, N


def get_cav_box(batch_data):
    """
    Args:
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    """

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
        agent_modality_list = batch_data['ego']['agent_modality_list']

    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        agent_modality_list = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
            agent_modality_list.append(cav_data['modality_name'])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()

        

    extent = [0.2, 0.2, 0.2]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_np = np.concatenate(box_list, axis=0)


    return cav_box_np, agent_modality_list