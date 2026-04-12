
# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
from opencood.models.pose_error_correction.boundalign_prm import box_matching
from opencood.utils import box_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')
import copy
def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=1,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    # 智能体数量
    parser.add_argument('--agent_num', type=int, default=2, help='number of agents to simulate')
    opt = parser.parse_args()
    return opt

def load_single_model(opt,single_idx):
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 
    opt_single = copy.deepcopy(opt)
    opt_single.model_dir = os.path.join(opt_single.model_dir, f'single{single_idx}')
    hypes = yaml_utils.load_yaml(None, opt_single)
    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)
        
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print(f'Creating Model_single{single_idx}')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading Model from checkpoint{single_idx}')
    saved_path = opt_single.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch of single{single_idx}.")
    opt_single.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    # setting noise
    np.random.seed(303)
    # build dataset for each noise setting
    print(f'Dataset Building single{single_idx}')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    infer_info = opt.fusion_method + opt.note



    return data_loader, opencood_dataset,model


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    

    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    
    infer_info = opt.fusion_method + opt.note

    agent_num= opt.agent_num
    data_loader_single_list = []
    opencood_dataset_single_list = []
    model_single_list = []
    for single_idx in range(1, agent_num+1):
        print(f"Loading single{single_idx} model and dataset...")
        data_loader_single, opencood_dataset_single, model_single = load_single_model(opt, single_idx)
        data_loader_single_list.append(data_loader_single)
        opencood_dataset_single_list.append(opencood_dataset_single)
        model_single_list.append(model_single)
    single_loader_iters = [iter(dl) for dl in data_loader_single_list]
    exit_i=0
    for i, batch_data in enumerate(data_loader):
        # if(exit_i>1):
        #     break
        # exit_i=exit_i+1
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            infer_result_single_list = []
            for idx in range(agent_num):
                batch_data_single = train_utils.to_device(next(single_loader_iters[idx]), device)
                infer_result_single = inference_utils.inference_intermediate_fusion(batch_data_single,
                                                                model_single_list[idx],
                                                                opencood_dataset_single_list[idx])
                infer_result_single_list.append(infer_result_single)
            # print(infer_result_single_list[0].keys())
            # print(infer_result_single_list[1].keys())
            # print(infer_result_single_list[2].keys())
            # print(infer_result_single_list[0]['pred_box_tensor'].shape)
            # print(infer_result_single_list[1]['pred_box_tensor'].shape)
            # print(infer_result_single_list[2]['pred_box_tensor'].shape)


            batch_data = train_utils.to_device(batch_data, device)
            pairwise=batch_data['ego']['pairwise_t_matrix']
            # print("pairwise shape:", pairwise.shape)

            pose_correct_matrix_list = []
            for single_idx in range(agent_num):
                if(single_idx==0):
                    pose_correct_matrix_list.append(np.eye(4, dtype=np.float32))
                    continue
                else:
                    # box_ref = infer_result_single_list[0]['pred_box_tensor']
                    box_ref = infer_result_single_list[0]['pred_box_tensor']
                    score_ref = infer_result_single_list[0]['pred_score']
                    box_cur = infer_result_single_list[single_idx]['pred_box_tensor']
                    score_cur = infer_result_single_list[single_idx]['pred_score']

                    # 任一方无框，直接跳过该agent校正
                    if box_ref is None or box_cur is None or score_ref is None or score_cur is None:
                        pose_correct_matrix_list.append(np.eye(4, dtype=np.float32))
                        continue
                    transformation_matrix = pairwise[0, single_idx, 0].to(box_ref.device, dtype=box_ref.dtype)
                    # pred_box_tensor_single1 = box_utils.project_box3d(infer_result_single_list[single_idx]['pred_box_tensor'], transformation_matrix)
                    pred_box_tensor_single1 = infer_result_single_list[single_idx]['pred_box_tensor']
                    pose_correct_matrix, match_info = box_matching(
                        infer_result_single_list[0]['pred_box_tensor'], # using single1 as reference
                        infer_result_single_list[0]['pred_score'],
                        infer_result_single_list[single_idx]['pred_box_tensor'],
                        infer_result_single_list[single_idx]['pred_score'],
                    )
                    # pose_correct_matrix_list.append(pose_correct_matrix)
                    # vis_root = "opencood/log/vis_feature"
                    # vis_name = f"bev_pair_frame{i:05d}_agent{single_idx}.png"
                    # vis_path = os.path.join(vis_root, vis_name)

                    # save_two_box_sets_bev(
                    # boxes_a=infer_result_single_list[0]["pred_box_tensor"],
                    # boxes_b=pred_box_tensor_single1,
                    # save_path=vis_path,
                    # color_a="red",
                    # color_b="lime",
                    # label_a="infer_result_single_list[0]['pred_box_tensor']",
                    # label_b="pred_box_tensor_single1"
                    # )
            print("pose_correct_matrix_list:", pose_correct_matrix_list)
            # 位姿校正
            new_pairwise = apply_pose_correction_to_pairwise(
            pairwise=pairwise,
            pose_correct_matrix_list=pose_correct_matrix_list,
            correction_is_delta=True
            )
            batch_data['ego']['pairwise_t_matrix'] = new_pairwise


            

            # print("batch keys:", batch_data['ego'].keys())

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
        torch.cuda.empty_cache()

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)


def apply_pose_correction_to_pairwise(pairwise, pose_correct_matrix_list, correction_is_delta=True):
    """
    pairwise: torch.Tensor, shape [1, L, L, 4, 4], pairwise[0,i,j] = T_{j<-i}
    pose_correct_matrix_list: list of (4,4) numpy/torch, index=i 对应 agent i 的校正矩阵
    correction_is_delta=True:
        T0i_new = D_i @ T0i_old
    correction_is_delta=False:
        T0i_new = D_i  (D_i 本身就是新 T_{0<-i})
    """
    assert pairwise.ndim == 5 and pairwise.shape[0] == 1
    L = int(pairwise.shape[1])
    device = pairwise.device
    dtype = pairwise.dtype

    # 1) 取旧的 T_{0<-i}
    T0_old = [pairwise[0, i, 0] for i in range(L)]

    # 2) 构建新的 T_{0<-i}
    T0_new = []
    for i in range(L):
        if i < len(pose_correct_matrix_list):
            D = pose_correct_matrix_list[i]
            if not torch.is_tensor(D):
                D = torch.tensor(D, device=device, dtype=dtype)
            else:
                D = D.to(device=device, dtype=dtype)
        else:
            D = torch.eye(4, device=device, dtype=dtype)

        if correction_is_delta:
            T_new_i = D @ T0_old[i]
        else:
            T_new_i = D

        T0_new.append(T_new_i)

    # ego 约束
    T0_new[0] = torch.eye(4, device=device, dtype=dtype)

    # 3) 重建 pairwise: T_{j<-i} = inv(T_{0<-j}) @ T_{0<-i}
    new_pair = torch.eye(4, device=device, dtype=dtype).view(1, 1, 1, 4, 4).repeat(1, L, L, 1, 1)
    for i in range(L):
        for j in range(L):
            if i == j:
                new_pair[0, i, j] = torch.eye(4, device=device, dtype=dtype)
            else:
                new_pair[0, i, j] = torch.linalg.solve(T0_new[j], T0_new[i])

    return new_pair


def _to_numpy_boxes(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)

def save_two_box_sets_bev(
    boxes_a,
    boxes_b,
    save_path,
    color_a="red",
    color_b="cyan",
    label_a="single1_ref",
    label_b="single_cur_proj",
    xlim=None,
    ylim=None,
    linewidth=1.2
):
    a = _to_numpy_boxes(boxes_a)
    b = _to_numpy_boxes(boxes_b)

    if a is None and b is None:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.3)

    def draw_set(boxes, color, label):
        if boxes is None or len(boxes) == 0:
            return
            
        # boxes: (N, 8, 3), BEV只取xy
        for k in range(boxes.shape[0]):
            # 取前4个角点闭合成多边形（在OpenCOOD中可用于BEV轮廓）
            poly = boxes[k, :4, :2]
            poly = np.concatenate([poly, poly[:1]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=linewidth, alpha=0.9)
            
        # 只给一条legend（缩进应当与for平齐，放在循环外部）
        ax.plot([], [], color=color, linewidth=linewidth, label=label)

    draw_set(a, color_a, label_a)
    draw_set(b, color_b, label_b)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.legend(loc="upper right")
    
    # 确保保存路径的文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == '__main__':
    main()




# # -*- coding: utf-8 -*-
# # Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# # License: TDG-Attribution-NonCommercial-NoDistrib

# import argparse
# import copy
# import importlib
# import os
# import re
# from collections import OrderedDict

# import numpy as np
# import torch
# from scipy.optimize import linear_sum_assignment
# from torch.utils.data import DataLoader

# import opencood.hypes_yaml.yaml_utils as yaml_utils
# from opencood.data_utils.datasets import build_dataset
# from opencood.tools import train_utils, inference_utils
# from opencood.utils import eval_utils
# from opencood.utils.common_utils import update_dict
# from opencood.visualization import simple_vis

# torch.multiprocessing.set_sharing_strategy("file_system")


# def test_parser():
#     parser = argparse.ArgumentParser(description="Single-first pose-corrected collaborative inference")
#     parser.add_argument("--model_dir", type=str, required=True,
#                         help="Collaborative model dir (contains config.yaml + checkpoints)")
#     parser.add_argument("--fusion_method", type=str, default="intermediate",
#                         help="late / early / intermediate / no / no_w_uncertainty / single")
#     parser.add_argument("--save_vis_interval", type=int, default=40)
#     parser.add_argument("--save_npy", action="store_true")
#     parser.add_argument("--range", type=str, default="102.4,102.4")
#     parser.add_argument("--no_score", action="store_true")
#     parser.add_argument("--note", default="", type=str)

#     # multi-single settings
#     parser.add_argument("--single_subdirs", type=str, default="single1,single2",
#                         help="comma separated subdirs under model_dir")
#     parser.add_argument("--single_agent_ids", type=str, default="1,2",
#                         help="comma separated agent ids for single_subdirs; align by index")
#     parser.add_argument("--single_default_ckpt", type=str, default="",
#                         help="optional checkpoint file name in each single dir, e.g. net_epoch_bestval_at37.pth")
#     parser.add_argument("--pose_match_radius", type=float, default=3.0)
#     parser.add_argument("--pose_min_pairs", type=int, default=3)
#     parser.add_argument("--pose_iter", type=int, default=2)
#     parser.add_argument("--max_samples", type=int, default=-1,
#                         help="debug only; -1 means all")

#     return parser.parse_args()


# def _parse_csv_int(s):
#     if s is None or len(s.strip()) == 0:
#         return []
#     return [int(x.strip()) for x in s.split(",") if len(x.strip()) > 0]


# def _parse_csv_str(s):
#     if s is None or len(s.strip()) == 0:
#         return []
#     return [x.strip() for x in s.split(",") if len(x.strip()) > 0]


# def _to_numpy_box_centers_xy(box_tensor):
#     # box_tensor: [N, 8, 3]
#     if box_tensor is None:
#         return np.zeros((0, 2), dtype=np.float32)
#     if torch.is_tensor(box_tensor):
#         box_np = box_tensor.detach().cpu().numpy()
#     else:
#         box_np = box_tensor
#     if box_np.shape[0] == 0:
#         return np.zeros((0, 2), dtype=np.float32)
#     # top face 4 corners mean as center
#     return box_np[:, :4, :2].mean(axis=1).astype(np.float32)


# # def _build_agent_input_from_ego(ego_data, agent_idx, num_agents):
# #     """
# #     Generic tensor slicer:
# #     - tensors with first dim == num_agents -> slice one agent
# #     - record_len -> set to [1]
# #     - pairwise_t_matrix -> keep 1x1 identity
# #     """
# #     agent_data = {}

# #     for k, v in ego_data.items():
# #         if k == "record_len":
# #             if torch.is_tensor(v):
# #                 agent_data[k] = torch.tensor([1], dtype=v.dtype, device=v.device)
# #             else:
# #                 agent_data[k] = [1]
# #             continue

# #         if k == "pairwise_t_matrix":
# #             dev = v.device if torch.is_tensor(v) else "cpu"
# #             dtype = v.dtype if torch.is_tensor(v) else torch.float32
# #             eye = torch.eye(4, device=dev, dtype=dtype).view(1, 1, 1, 4, 4)
# #             agent_data[k] = eye
# #             continue

# #         if torch.is_tensor(v):
# #             if v.dim() > 0 and int(v.shape[0]) == int(num_agents):
# #                 agent_data[k] = v[agent_idx:agent_idx + 1]
# #             else:
# #                 agent_data[k] = v
# #         elif isinstance(v, dict):
# #             agent_data[k] = _build_agent_input_from_ego(v, agent_idx, num_agents)
# #         else:
# #             # list / scalar / string keep as is
# #             agent_data[k] = v

# #     return agent_data

# def _build_agent_input_from_ego(ego_data, agent_idx, num_agents):
#     """
#     Generic tensor slicer:
#     - tensors with first dim == num_agents -> slice one agent
#     - modality_name -> extract agent-specific modality (if list)
#     - record_len -> set to [1]
#     - pairwise_t_matrix -> keep 1x1 identity
#     """
#     agent_data = {}

#     for k, v in ego_data.items():
#         if k == "modality_name":
#             # modality_name may be a list like ['m1', 'm2', 'm3']
#             # extract the one for this agent
#             if isinstance(v, list) and len(v) == num_agents:
#                 agent_data[k] = [v[agent_idx]]  # wrap in list to keep format
#             else:
#                 agent_data[k] = v
#             continue

#         if k == "record_len":
#             if torch.is_tensor(v):
#                 agent_data[k] = torch.tensor([1], dtype=v.dtype, device=v.device)
#             else:
#                 agent_data[k] = [1]
#             continue

#         if k == "pairwise_t_matrix":
#             dev = v.device if torch.is_tensor(v) else "cpu"
#             dtype = v.dtype if torch.is_tensor(v) else torch.float32
#             eye = torch.eye(4, device=dev, dtype=dtype).view(1, 1, 1, 4, 4)
#             agent_data[k] = eye
#             continue

#         if torch.is_tensor(v):
#             if v.dim() > 0 and int(v.shape[0]) == int(num_agents):
#                 agent_data[k] = v[agent_idx:agent_idx + 1]
#             else:
#                 agent_data[k] = v
#         elif isinstance(v, dict):
#             agent_data[k] = _build_agent_input_from_ego(v, agent_idx, num_agents)
#         else:
#             # list / scalar / string keep as is
#             agent_data[k] = v

#     return agent_data

# def _decode_single_boxes(dataset, model_out, anchor_box, t_agent_to_ego, device):
#     """
#     Decode one single model output by dataset post_processor.
#     Returns pred_box_tensor, pred_score in ego frame if t_agent_to_ego is not identity.
#     """
#     if t_agent_to_ego is None:
#         t_agent_to_ego = torch.eye(4, device=device, dtype=torch.float32)

#     single_data_dict = OrderedDict({
#         "ego": {
#             "anchor_box": anchor_box,
#             "transformation_matrix": t_agent_to_ego,
#             "transformation_matrix_clean": t_agent_to_ego
#         }
#     })

#     single_out = {
#         "cls_preds": model_out["cls_preds"],
#         "reg_preds": model_out["reg_preds"]
#     }
#     if "dir_preds" in model_out:
#         single_out["dir_preds"] = model_out["dir_preds"]

#     output_dict = OrderedDict({"ego": single_out})

#     pred_box_tensor, pred_score = dataset.post_processor.post_process(single_data_dict, output_dict)
#     return pred_box_tensor, pred_score


# def _apply_se2(T, pts_xy):
#     # T: [4,4], pts_xy: [N,2]
#     if pts_xy.shape[0] == 0:
#         return pts_xy
#     ones = np.ones((pts_xy.shape[0], 1), dtype=np.float32)
#     xyz1 = np.concatenate([pts_xy, np.zeros((pts_xy.shape[0], 1), dtype=np.float32), ones], axis=1)  # [N,4]
#     out = (T @ xyz1.T).T
#     return out[:, :2]


# def _estimate_se2_from_correspondence(src_xy, dst_xy):
#     """
#     Solve dst ~= R * src + t via SVD in 2D.
#     src_xy: [N,2], dst_xy: [N,2]
#     """
#     assert src_xy.shape[0] == dst_xy.shape[0]
#     n = src_xy.shape[0]
#     if n < 2:
#         return None

#     src_mean = src_xy.mean(axis=0, keepdims=True)
#     dst_mean = dst_xy.mean(axis=0, keepdims=True)

#     src_c = src_xy - src_mean
#     dst_c = dst_xy - dst_mean

#     H = src_c.T @ dst_c
#     U, _, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T

#     # reflection fix
#     if np.linalg.det(R) < 0:
#         Vt[1, :] *= -1
#         R = Vt.T @ U.T

#     t = dst_mean.reshape(2, 1) - R @ src_mean.reshape(2, 1)

#     T = np.eye(4, dtype=np.float32)
#     T[0:2, 0:2] = R.astype(np.float32)
#     T[0:2, 3:4] = t.astype(np.float32)

#     return T


# def _refine_agent_pose_from_points(src_local_xy, target_ego_xy, T_init,
#                                    match_radius=3.0, min_pairs=3, max_iter=2):
#     """
#     Custom iterative matching + rigid fit.
#     - src_local_xy: agent-local detections [N,2]
#     - target_ego_xy: global target detections in ego frame [M,2]
#     - T_init: initial T_{0<-i}, shape [4,4]
#     """
#     if src_local_xy.shape[0] == 0 or target_ego_xy.shape[0] == 0:
#         return T_init, 0

#     T = T_init.copy()
#     best_pairs = 0

#     for _ in range(max_iter):
#         src_in_ego = _apply_se2(T, src_local_xy)
#         if src_in_ego.shape[0] == 0 or target_ego_xy.shape[0] == 0:
#             break

#         dist = np.linalg.norm(src_in_ego[:, None, :] - target_ego_xy[None, :, :], axis=2)
#         r_idx, c_idx = linear_sum_assignment(dist)

#         valid = dist[r_idx, c_idx] < match_radius
#         r_idx = r_idx[valid]
#         c_idx = c_idx[valid]

#         if r_idx.shape[0] < min_pairs:
#             break

#         src_match = src_local_xy[r_idx]
#         dst_match = target_ego_xy[c_idx]

#         T_new = _estimate_se2_from_correspondence(src_match, dst_match)
#         if T_new is None:
#             break

#         T = T_new
#         best_pairs = int(r_idx.shape[0])

#     return T, best_pairs


# def _build_pairwise_from_t0i(t0i_list, max_cav, device, dtype):
#     """
#     Build pairwise matrix from T_{0<-i}.
#     pairwise[0, i, j] = T_{j<-i} = inv(T_{0<-j}) @ T_{0<-i}
#     """
#     N = len(t0i_list)
#     pairwise = torch.eye(4, device=device, dtype=dtype).view(1, max_cav, max_cav, 4, 4).clone()

#     # torch inverse for better numeric consistency on device
#     T0 = []
#     for i in range(N):
#         T0.append(torch.tensor(t0i_list[i], device=device, dtype=dtype))

#     for i in range(N):
#         for j in range(N):
#             if i == j:
#                 pairwise[0, i, j] = torch.eye(4, device=device, dtype=dtype)
#             else:
#                 pairwise[0, i, j] = torch.linalg.solve(T0[j], T0[i])

#     return pairwise


# def _load_one_model_with_yaml(model_dir):
#     opt_tmp = argparse.Namespace(model_dir=model_dir)
#     hypes = yaml_utils.load_yaml(None, opt_tmp)
#     model = train_utils.create_model(hypes)
#     _, model = train_utils.load_saved_model(model_dir, model)
#     return hypes, model


# def _load_single_models(opt, device):
#     subdirs = _parse_csv_str(opt.single_subdirs)
#     agent_ids = _parse_csv_int(opt.single_agent_ids)

#     if len(subdirs) == 0:
#         raise ValueError("single_subdirs is empty")
#     if len(agent_ids) != len(subdirs):
#         raise ValueError("single_agent_ids length must equal single_subdirs length")

#     single_models = {}
#     single_hypes = {}

#     for sd, aid in zip(subdirs, agent_ids):
#         sdir = os.path.join(opt.model_dir, sd)
#         if not os.path.isdir(sdir):
#             raise FileNotFoundError("single model dir not found: {}".format(sdir))

#         h_s, m_s = _load_one_model_with_yaml(sdir)
#         if torch.cuda.is_available():
#             m_s.cuda()
#         m_s.eval()

#         single_models[aid] = m_s
#         single_hypes[aid] = h_s
#         print("Loaded single model: agent_id={} dir={}".format(aid, sdir))

#     return single_models, single_hypes


# def _run_single_stage_and_refine(batch_data, dataset, single_models, opt, device):
#     """
#     1) per-agent single detection in local frame
#     2) build global target from all projected detections
#     3) refine each T_{0<-i}
#     4) overwrite pairwise_t_matrix in batch_data
#     """
#     ego = batch_data["ego"]
#     record_len = ego["record_len"]
#     if torch.is_tensor(record_len):
#         num_agents = int(record_len[0].item())
#     else:
#         num_agents = int(record_len[0])

#     pairwise_old = ego["pairwise_t_matrix"][0]  # [L,L,4,4]
#     max_cav = int(pairwise_old.shape[0])

#     # init T_{0<-i}
#     t0i = []
#     for i in range(num_agents):
#         t0i.append(pairwise_old[i, 0].detach().cpu().numpy().astype(np.float32))

#     # local detections per agent
#     local_centers = {}
#     ego_projected_centers = {}

#     anchor_box = ego["anchor_box"]

#     for i in range(num_agents):
#         # choose model by agent id
#         if i not in single_models:
#             # no model for this agent, skip
#             local_centers[i] = np.zeros((0, 2), dtype=np.float32)
#             ego_projected_centers[i] = np.zeros((0, 2), dtype=np.float32)
#             continue

#         model_s = single_models[i]

#         one_agent_input = _build_agent_input_from_ego(ego, i, num_agents)
#         out = model_s(one_agent_input)

#         # decode in agent local frame
#         I4 = torch.eye(4, device=device, dtype=torch.float32)
#         pred_local, _ = _decode_single_boxes(dataset, out, anchor_box, I4, device)
#         c_local = _to_numpy_box_centers_xy(pred_local)
#         local_centers[i] = c_local

#         # project local -> ego with current T_{0<-i}
#         c_ego = _apply_se2(t0i[i], c_local)
#         ego_projected_centers[i] = c_ego

#     # global target in ego frame (exclude current agent when refining each agent)
#     refined_t0i = copy.deepcopy(t0i)
#     pair_count_dict = {}

#     for i in range(1, num_agents):  # usually refine non-ego
#         if i not in single_models:
#             pair_count_dict[i] = 0
#             continue

#         src_local = local_centers[i]
#         if src_local.shape[0] == 0:
#             pair_count_dict[i] = 0
#             continue

#         target_list = []
#         for j in range(num_agents):
#             if j == i:
#                 continue
#             c = ego_projected_centers.get(j, None)
#             if c is not None and c.shape[0] > 0:
#                 target_list.append(c)

#         if len(target_list) == 0:
#             pair_count_dict[i] = 0
#             continue

#         target_ego = np.concatenate(target_list, axis=0)
#         T_new, n_pairs = _refine_agent_pose_from_points(
#             src_local_xy=src_local,
#             target_ego_xy=target_ego,
#             T_init=t0i[i],
#             match_radius=opt.pose_match_radius,
#             min_pairs=opt.pose_min_pairs,
#             max_iter=opt.pose_iter,
#         )
#         refined_t0i[i] = T_new
#         pair_count_dict[i] = n_pairs

#     # keep ego as identity
#     refined_t0i[0] = np.eye(4, dtype=np.float32)

#     pairwise_new = _build_pairwise_from_t0i(
#         t0i_list=refined_t0i,
#         max_cav=max_cav,
#         device=ego["pairwise_t_matrix"].device,
#         dtype=ego["pairwise_t_matrix"].dtype,
#     )

#     batch_data["ego"]["pairwise_t_matrix"] = pairwise_new

#     return batch_data, pair_count_dict


# def main():
#     opt = test_parser()
#     assert opt.fusion_method in ["late", "early", "intermediate", "no", "no_w_uncertainty", "single"]

#     # collaborative hypes from model_dir
#     hypes = yaml_utils.load_yaml(None, opt)

#     if "heter" in hypes:
#         x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
#         y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
#         opt.note += "_{}_{}".format(x_max, y_max)

#         new_cav_range = [
#             x_min, y_min, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
#             x_max, y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5]
#         ]

#         hypes = update_dict(hypes, {
#             "cav_lidar_range": new_cav_range,
#             "lidar_range": new_cav_range,
#             "gt_range": new_cav_range
#         })

#         yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
#         parser_func = None
#         for name, func in yaml_utils_lib.__dict__.items():
#             if name == hypes["yaml_parser"]:
#                 parser_func = func
#                 break
#         if parser_func is not None:
#             hypes = parser_func(hypes)

#     hypes["validate_dir"] = hypes["test_dir"]
#     if "OPV2V" in hypes["test_dir"] or "v2xsim" in hypes["test_dir"]:
#         assert "test" in hypes["validate_dir"]

#     left_hand = True if ("OPV2V" in hypes["test_dir"] or "V2XSET" in hypes["test_dir"]) else False
#     print("Left hand visualizing: {}".format(left_hand))

#     if "box_align" in hypes.keys():
#         hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # load collaborative model
#     print("Creating collaborative model")
#     collab_model = train_utils.create_model(hypes)
#     print("Loading collaborative model from checkpoint")
#     resume_epoch, collab_model = train_utils.load_saved_model(opt.model_dir, collab_model)
#     print("Collaborative resume epoch: {}".format(resume_epoch))
#     opt.note += "_epoch{}".format(resume_epoch)
#     if torch.cuda.is_available():
#         collab_model.cuda()
#     collab_model.eval()

#     # load single models
#     print("Loading single models")
#     single_models, _ = _load_single_models(opt, device)

#     np.random.seed(303)

#     print("Dataset Building")
#     opencood_dataset = build_dataset(hypes, visualize=True, train=False)

#     data_loader = DataLoader(
#         opencood_dataset,
#         batch_size=1,
#         num_workers=4,
#         collate_fn=opencood_dataset.collate_batch_test,
#         shuffle=False,
#         pin_memory=False,
#         drop_last=False
#     )

#     result_stat = {
#         0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
#         0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
#         0.7: {"tp": [], "fp": [], "gt": 0, "score": []}
#     }

#     infer_info = opt.fusion_method + opt.note + "_singleFirstPoseRefine"

#     for i, batch_data in enumerate(data_loader):
#         if opt.max_samples > 0 and i >= opt.max_samples:
#             break

#         print("{}_{}".format(infer_info, i))
#         if batch_data is None:
#             continue

#         with torch.no_grad():
#             batch_data = train_utils.to_device(batch_data, device)

#             # stage-1: single inference + custom pose correction
#             batch_data, pair_count_dict = _run_single_stage_and_refine(
#                 batch_data=batch_data,
#                 dataset=opencood_dataset,
#                 single_models=single_models,
#                 opt=opt,
#                 device=device
#             )
#             print("pose refine pairs:", pair_count_dict)

#             # stage-2: collaborative inference with corrected pairwise
#             if opt.fusion_method == "late":
#                 infer_result = inference_utils.inference_late_fusion(batch_data, collab_model, opencood_dataset)
#             elif opt.fusion_method == "early":
#                 infer_result = inference_utils.inference_early_fusion(batch_data, collab_model, opencood_dataset)
#             elif opt.fusion_method == "intermediate":
#                 infer_result = inference_utils.inference_intermediate_fusion(batch_data, collab_model, opencood_dataset)
#             elif opt.fusion_method == "no":
#                 infer_result = inference_utils.inference_no_fusion(batch_data, collab_model, opencood_dataset)
#             elif opt.fusion_method == "no_w_uncertainty":
#                 infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data, collab_model, opencood_dataset)
#             elif opt.fusion_method == "single":
#                 infer_result = inference_utils.inference_no_fusion(batch_data, collab_model, opencood_dataset, single_gt=True)
#             else:
#                 raise NotImplementedError("unsupported fusion_method: {}".format(opt.fusion_method))

#             pred_box_tensor = infer_result["pred_box_tensor"]
#             gt_box_tensor = infer_result["gt_box_tensor"]
#             pred_score = infer_result["pred_score"]

#             eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
#             eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
#             eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

#             if opt.save_npy:
#                 npy_save_path = os.path.join(opt.model_dir, "npy")
#                 if not os.path.exists(npy_save_path):
#                     os.makedirs(npy_save_path)
#                 inference_utils.save_prediction_gt(
#                     pred_box_tensor,
#                     gt_box_tensor,
#                     batch_data["ego"]["origin_lidar"][0],
#                     i,
#                     npy_save_path
#                 )

#             if not opt.no_score:
#                 infer_result.update({"score_tensor": pred_score})

#             if getattr(opencood_dataset, "heterogeneous", False):
#                 cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
#                 infer_result.update({
#                     "cav_box_np": cav_box_np,
#                     "agent_modality_list": agent_modality_list
#                 })

#             if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
#                 vis_save_path_root = os.path.join(opt.model_dir, "vis_{}".format(infer_info))
#                 if not os.path.exists(vis_save_path_root):
#                     os.makedirs(vis_save_path_root)

#                 vis_save_path = os.path.join(vis_save_path_root, "bev_{:05d}.png".format(i))
#                 simple_vis.visualize(
#                     infer_result,
#                     batch_data["ego"]["origin_lidar"][0],
#                     hypes["postprocess"]["gt_range"],
#                     vis_save_path,
#                     method="bev",
#                     left_hand=left_hand
#                 )

#         torch.cuda.empty_cache()

#     _, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, infer_info)
#     print("Final AP50: {:.4f}, AP70: {:.4f}".format(ap50, ap70))


# if __name__ == "__main__":
#     main()

