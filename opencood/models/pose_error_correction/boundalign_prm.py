# from opencood.utils import box_utils
# import numpy as np
# from scipy.optimize import linear_sum_assignment
# def box_matching(pred_box_tensor_ego, pred_box_tensor_single_1, pred_score_tensor_single_1):
#     # 提取三位边界框的中心点和长宽(N,8,3）角点
#     pred_box_ego = pred_box_tensor_ego.cpu().numpy()
#     pred_box_single_1 = pred_box_tensor_single_1.cpu().numpy()
#     pred_score_single_1 = pred_score_tensor_single_1.cpu().numpy()
#     # Convert 8 corners to x, y, z, dx, dy, dz, yaw.
#     # yaw in radians
#     pred_box_ego_bev = box_utils.corner_to_center(pred_box_ego)  # (N,8,3)
#     pred_box_single_1_bev = box_utils.corner_to_center(pred_box_single_1)  # (N,8,3)

#     # 相同框进行匹配，求解single1到ego的变换矩阵



#     pass


# 可以，下面给你一版可直接用的实现。功能是：

# 把角点框转成 BEV 参数框

# 用匈牙利匹配找同一目标

# 用匹配点估计 single1 到 ego 的 2D 刚体变换

# 返回 4x4 变换矩阵和匹配信息

import numpy as np
from scipy.optimize import linear_sum_assignment
from opencood.utils import box_utils


def _to_numpy(x):
	if x is None:
		return None
	if hasattr(x, "detach"):
		x = x.detach()
	if hasattr(x, "cpu"):
		x = x.cpu()
	if hasattr(x, "numpy"):
		return x.numpy()
	return np.asarray(x)


def _wrap_angle(rad):
	return (rad + np.pi) % (2.0 * np.pi) - np.pi


def _weighted_rigid_transform_2d(src_xy, dst_xy, weights):
	"""
	求解 dst ~= R * src + t
	src_xy: (N,2) single坐标
	dst_xy: (N,2) ego坐标
	weights: (N,)
	"""
	w = np.asarray(weights, dtype=np.float64).reshape(-1)
	w = np.maximum(w, 1e-6)
	w = w / (w.sum() + 1e-12)

	src = np.asarray(src_xy, dtype=np.float64)
	dst = np.asarray(dst_xy, dtype=np.float64)

	src_mean = np.sum(src * w[:, None], axis=0, keepdims=True)
	dst_mean = np.sum(dst * w[:, None], axis=0, keepdims=True)

	src_c = src - src_mean
	dst_c = dst - dst_mean

	H = (src_c * w[:, None]).T @ dst_c
	U, _, Vt = np.linalg.svd(H)
	R = Vt.T @ U.T

	# 避免反射
	if np.linalg.det(R) < 0:
		Vt[-1, :] *= -1
		R = Vt.T @ U.T

	t = dst_mean.reshape(2, 1) - R @ src_mean.reshape(2, 1)

	T = np.eye(4, dtype=np.float32)
	T[0:2, 0:2] = R.astype(np.float32)
	T[0:2, 3:4] = t.astype(np.float32)
	return T


def box_matching(
	pred_box_tensor_ego,
	pred_score_tensor_ego,
	pred_box_tensor_single_1,
	pred_score_tensor_single_1,
	score_thr=0.1,
	max_center_dist=8.0,
	w_center=1.0,
	w_size=0.4,
	w_yaw=0.2,
):
	"""
	输入:
	pred_box_tensor_ego: (Ne,8,3)
	pred_score_tensor_ego: (Ne,)
	pred_box_tensor_single_1: (Ns,8,3)
	pred_score_tensor_single_1: (Ns,)
	输出:
	T_single_to_ego: (4,4) float32
	match_info: dict
	"""

	# 默认返回单位阵
	T_identity = np.eye(4, dtype=np.float32)

	# 1) 转 numpy
	pred_box_ego = _to_numpy(pred_box_tensor_ego)
	pred_score_ego = _to_numpy(pred_score_tensor_ego)
	pred_box_single = _to_numpy(pred_box_tensor_single_1)
	pred_score_single = _to_numpy(pred_score_tensor_single_1)

	if pred_box_ego is None or pred_score_ego is None or pred_box_single is None or pred_score_single is None:
		return T_identity, {"num_match": 0, "reason": "none_input"}

	pred_score_ego = np.asarray(pred_score_ego).reshape(-1)
	pred_score_single = np.asarray(pred_score_single).reshape(-1)

	if pred_box_ego.shape[0] == 0 or pred_box_single.shape[0] == 0:
		return T_identity, {"num_match": 0, "reason": "empty_box"}

	if pred_score_ego.shape[0] == 0 or pred_score_single.shape[0] == 0:
		return T_identity, {"num_match": 0, "reason": "empty_score"}

	if pred_box_ego.shape[0] != pred_score_ego.shape[0]:
		return T_identity, {"num_match": 0, "reason": "ego_box_score_mismatch"}

	if pred_box_single.shape[0] != pred_score_single.shape[0]:
		return T_identity, {"num_match": 0, "reason": "single_box_score_mismatch"}

	# 2) 角点 -> 参数框 (N,7): x,y,z,l,w,h,yaw
	ego_bev = box_utils.corner_to_center(pred_box_ego, order="lwh")
	single_bev = box_utils.corner_to_center(pred_box_single, order="lwh")

	# 3) 双边分数筛选
	keep_single = pred_score_single >= score_thr
	keep_ego = pred_score_ego >= score_thr
	if keep_single.sum() == 0:
		return T_identity, {"num_match": 0, "reason": "all_single_low_score"}
	if keep_ego.sum() == 0:
		return T_identity, {"num_match": 0, "reason": "all_ego_low_score"}

	single_bev = single_bev[keep_single]
	single_score = pred_score_single[keep_single]
	ego_bev = ego_bev[keep_ego]
	ego_score = pred_score_ego[keep_ego]

	if ego_bev.shape[0] == 0 or single_bev.shape[0] == 0:
		return T_identity, {"num_match": 0, "reason": "empty_after_filter"}

	ego_xy = ego_bev[:, 0:2]
	ego_lw = ego_bev[:, 3:5]
	ego_yaw = ego_bev[:, 6]

	single_xy = single_bev[:, 0:2]
	single_lw = single_bev[:, 3:5]
	single_yaw = single_bev[:, 6]

	# 4) 构建代价矩阵
	# center 距离
	center_dist = np.linalg.norm(single_xy[:, None, :] - ego_xy[None, :, :], axis=2)  # (Ns,Ne)

	# size 差异
	size_dist = np.linalg.norm(single_lw[:, None, :] - ego_lw[None, :, :], axis=2)  # (Ns,Ne)

	# yaw 差异
	
	# yaw_diff = np.abs(_wrap_angle(single_yaw[:, None] - ego_yaw[None, :]))  # (Ns,Ne)
	yaw_diff_raw = np.abs(_wrap_angle(single_yaw[:, None] - ego_yaw[None, :]))
	yaw_diff = np.minimum(yaw_diff_raw, np.pi - yaw_diff_raw)

	cost = w_center * center_dist + w_size * size_dist + w_yaw * yaw_diff

	# 高置信度匹配给予小幅奖励（降低代价）
	conf_bonus = 0.2 * (single_score[:, None] + ego_score[None, :])
	cost = cost - conf_bonus

	# 门控: 超过最大中心距离直接禁用
	cost[center_dist > max_center_dist] = 1e6

	# 5) 匈牙利匹配
	row_ind, col_ind = linear_sum_assignment(cost)

	valid = cost[row_ind, col_ind] < 1e5
	row_ind = row_ind[valid]
	col_ind = col_ind[valid]

	if row_ind.shape[0] < 2:
		# 少于2对，无法稳定估计旋转，退化返回单位阵
		return T_identity, {
			"num_match": int(row_ind.shape[0]),
			"reason": "not_enough_match",
			"single_indices": row_ind.tolist(),
			"ego_indices": col_ind.tolist(),
		}

	# 6) 用匹配点估计 single->ego 变换
	src_xy = single_xy[row_ind]  # single
	dst_xy = ego_xy[col_ind]  # ego
	weights = single_score[row_ind] * ego_score[col_ind]

	T_single_to_ego = _weighted_rigid_transform_2d(src_xy, dst_xy, weights)

	return T_single_to_ego, {
		"num_match": int(row_ind.shape[0]),
		"reason": "ok",
		"single_indices": row_ind.tolist(),
		"ego_indices": col_ind.tolist(),
		"matched_weight_mean": float(weights.mean()),
		"mean_center_dist": float(center_dist[row_ind, col_ind].mean()),
	}