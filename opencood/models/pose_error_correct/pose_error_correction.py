import torch
import math
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

def create_delta_affine(dx, dy, dtheta, H, W, device):
    """
    构建相对位姿偏差的仿射变换矩阵
    """
    tx = 2.0 * dx / max(W - 1, 1)
    ty = 2.0 * dy / max(H - 1, 1)
    theta_rad = math.radians(dtheta)
    
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    
    affine = torch.tensor([
        [cos_t, -sin_t, tx],[sin_t,  cos_t, ty]
    ], dtype=torch.float32, device=device)
    
    return affine

def compose_affine(delta_2x3, base_2x3):
    """ 
    【关键修复1】矩阵乘法顺序必须是 base @ delta！
    因为 delta 是在 Ego (目标) 坐标系下的纠正，必须先纠正 Ego 的网格，再去用 base 映射到 Neighbor
    """
    device = base_2x3.device
    delta_3x3 = torch.eye(3, device=device)
    delta_3x3[:2, :3] = delta_2x3
    
    base_3x3 = torch.eye(3, device=device)
    base_3x3[:2, :3] = base_2x3
    
    # 修复：将 delta 乘在右边
    composed = torch.mm(base_3x3, delta_3x3)
    return composed[:2, :3]

class CartographerPoseCorrector(torch.nn.Module):
    """
    基于 Cartographer 思想的分层并行搜索即插即用模块
    """
    # 【关键修复3】调大搜索范围。trans_range=30 意味着能搜索 30*0.4m = 12米的误差！
    def __init__(self, trans_range=30.0, rot_range=15.0, coarse_step=2.0, fine_step=0.5):
        super().__init__()
        self.trans_range = trans_range 
        self.rot_range = rot_range     
        self.coarse_step = coarse_step 
        self.fine_step = fine_step     

    @torch.no_grad()
    def refine_pair(self, ego_score, nbr_score, base_affine, align_corners=False):
        _, _, H, W = ego_score.shape
        device = ego_score.device

        # 【关键修复2】过滤掉背景，只保留置信度大于 0.3 的特征点参与计算！
        # 彻底防止匹配到相机的黑边盲区上
        ego_score_clean = torch.where(ego_score > 0.3, ego_score, torch.zeros_like(ego_score))
        nbr_score_clean = torch.where(nbr_score > 0.3, nbr_score, torch.zeros_like(nbr_score))

        def _search_grid(dx_range, dy_range, dr_range):
            grid_dx, grid_dy, grid_dr = torch.meshgrid(dx_range, dy_range, dr_range, indexing='ij')
            candidates = torch.stack([grid_dx.flatten(), grid_dy.flatten(), grid_dr.flatten()], dim=1)
            N_cand = candidates.shape[0]

            candidate_affines = torch.zeros((N_cand, 2, 3), device=device)
            for i in range(N_cand):
                delta_aff = create_delta_affine(candidates[i,0], candidates[i,1], candidates[i,2], H, W, device)
                candidate_affines[i] = compose_affine(delta_aff, base_affine)

            # 注意这里用的是 clean (被清理过背景) 的图做计算
            nbr_score_batch = nbr_score_clean.expand(N_cand, 1, H, W)
            nbr_warped_batch = warp_affine_simple(nbr_score_batch, candidate_affines, (H, W), align_corners=align_corners)

            ego_score_batch = ego_score_clean.expand(N_cand, 1, H, W)
            overlap_scores = (ego_score_batch * nbr_warped_batch).sum(dim=[1, 2, 3])

            best_idx = torch.argmax(overlap_scores)
            
            # 【保险机制】：如果所有网格的重叠度都是0（没匹配上），返回原始变换
            max_score = overlap_scores[best_idx]
            if max_score <= 1e-5:
                return base_affine, torch.tensor([0.0, 0.0, 0.0], device=device)

            return candidate_affines[best_idx], candidates[best_idx]

        # === 阶段1：粗匹配 ===
        dx_coarse = torch.arange(-self.trans_range, self.trans_range + 1e-3, self.coarse_step, device=device)
        dy_coarse = torch.arange(-self.trans_range, self.trans_range + 1e-3, self.coarse_step, device=device)
        dr_coarse = torch.arange(-self.rot_range, self.rot_range + 1e-3, self.coarse_step, device=device)
        
        _, best_coarse_params = _search_grid(dx_coarse, dy_coarse, dr_coarse)

        # 如果全图找不到共同的高亮车辆（返回了0,0,0），直接退出，避免破坏原位姿
        if torch.all(best_coarse_params == 0.0):
            return base_affine

        # === 阶段2：精匹配 ===
        c_x, c_y, c_r = best_coarse_params
        dx_fine = torch.arange(c_x - self.coarse_step, c_x + self.coarse_step + 1e-3, self.fine_step, device=device)
        dy_fine = torch.arange(c_y - self.coarse_step, c_y + self.coarse_step + 1e-3, self.fine_step, device=device)
        dr_fine = torch.arange(c_r - self.coarse_step, c_r + self.coarse_step + 1e-3, self.fine_step, device=device)

        best_fine_affine, _ = _search_grid(dx_fine, dy_fine, dr_fine)

        return best_fine_affine

    @torch.no_grad()
    def forward(self, occ_map, record_len, affine_matrix, align_corners=False):
        refined_affine = affine_matrix.clone()
        split_occ = regroup(occ_map, record_len)
        B = len(split_occ)

        for b in range(B):
            N = int(record_len[b])
            if N <= 1:
                continue
            
            ego_occ = split_occ[b][0:1] 
            for n in range(1, N):
                nbr_occ = split_occ[b][n:n+1]
                base_affine = refined_affine[b, 0, n]
                
                refined_affine[b, 0, n] = self.refine_pair(
                    ego_occ, nbr_occ, base_affine, align_corners
                )

        return refined_affine