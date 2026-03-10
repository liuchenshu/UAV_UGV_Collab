import json
import numpy as np
import open3d as o3d
from PIL import Image
# ---------lidar、pose、calibration、label的读取函数，适用于griffin数据集------------------------------
# grffin图片格式为RGBA，我们只需要RGB通道，因此在读取时使用convert('RGB')转换为RGB格式
def load_camera_data_griffin(camera_files, preload=True):
    """
    Args:
        camera_files: list, 
            store camera path
        shape : tuple
            (width, height), resize the image, and overcoming the lazy loading.
    Returns:
        camera_data_list: list,
            list of Image, RGB order
    """
    camera_data_list = []
    for camera_file in camera_files:
        camera_data = Image.open(camera_file).convert('RGB')
        if preload:
            camera_data = camera_data.copy()
        camera_data_list.append(camera_data)
    return camera_data_list
# 原始opencood框架使用pypcd读取点云文件，适用于dairv2x的pcd文件，griffin数据集为.ply点云文件
# 这里使用o3d.t读取.ply的坐标与强度，key键为在点云文件的注释，例如：
# ply
# format ascii 1.0
# element vertex 2929
# property float32 x
# property float32 y
# property float32 z
# property float32 I
# end_header
# -0.4505 -0.1243 0.1252 0.9981
# xyz合并为positions，强度I单独存储为I
def load_lidar_ply_griffin(file_path):
    pcd = o3d.t.io.read_point_cloud(file_path)
    lidarpoints = pcd.point['positions'].numpy()  
    intensity = pcd.point['I'].numpy() 
    # distances = np.linalg.norm(lidarpoints, axis=1)
    # threshold = 100.0  # 设置距离阈值
    # filtered_points = lidarpoints[distances < threshold]
    data=np.hstack((lidarpoints, intensity))  # 将强度信息添加到点云数据中
    return data

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_pose_json_griffin(file_path):
    pose = load_json(file_path)
    t = np.array([pose.get('x', 0.0), pose.get('y', 0.0), pose.get('z', 0.0)], dtype=float)
    rpy = np.array([pose.get('roll', 0.0), pose.get('pitch', 0.0), pose.get('yaw', 0.0)], dtype=float)
    # dairv2x需要的是角度值
    rpy = np.deg2rad(rpy)
    x, y, z = t
    roll, pitch, yaw = rpy

    return [x, y, z, roll, yaw, pitch]


def load_calibration_json_griffin(path):
    with open(path, 'r', encoding='utf-8') as f:
        calib = json.load(f)
    instrinsic = np.array(calib['intrinsic'], dtype=np.float32)
    extrinsic = np.array(calib['extrinsic'], dtype=np.float32)
    distortion = np.array(calib['distortion'], dtype=np.float32)
    return instrinsic, extrinsic, distortion

def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 12:
        return None
    cls = parts[0]
    x, y, z = map(float, parts[1:4])
    dx, dy, dz = map(float, parts[4:7])
    rx, ry, rz = map(float, parts[7:10])
    #转换为弧度
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    obj_id = int(parts[10])
    score = float(parts[11])
    return {
        'class': cls,
        'x': x, 'y': y, 'z': z,
        'dx': dx, 'dy': dy, 'dz': dz,
        'rx': rx, 'ry': ry, 'rz': rz,
        'id': obj_id, 'score': score
    }

def load_label_txt(path):
    objs = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('//') or ln.startswith('#'):
                continue
            obj = parse_label_line(ln)
            #过滤score<0.8的标签
            if obj and obj['score'] >= 0.8:
                objs.append(obj)
    return objs

#标签转换为dairv2x格式的标签字典
# "type": "car",
# "3d_dimensions": {"h": 1.567295, "w": 2.116977, "l": 4.78775},
#  "3d_location": {"x": 2649.354689288955, "y": 1723.1646547804826, "z": 20.556637042945873}, 
# "rotation": 0.05378114,
# 同时dairv2x协同数据集需要世界坐标系下的八个角点数据
# "world_8_points": 
# [[2699.0250672305233, 1663.379419095183, 19.88840887956328], 
# [2697.5283606190656, 1662.4247065439965, 19.862130372795534], 
# [2695.167344471905, 1666.1256006473477, 19.87957528081578], 
# [2696.6640510833627, 1667.0803131985342, 19.905853787583524], 
# [2699.009439598601, 1663.362326639559, 21.39946740681461], 
# [2697.5127329871434, 1662.4076140883728, 21.373188900046863], 
# [2695.1517168399832, 1666.1085081917238, 21.390633808067108], 
# [2696.6484234514405, 1667.0632207429103, 21.416912314834853]
def griffin_label_to_dairv2x_label(label):
    dairv2x_label=[]
    for obj in label:
        dairv2x_obj=obj.copy()
        # griffin的标签类别与dairv2x的标签类别不完全一致，这里做一个简单的映射
        # grffin标签类别都是小写，而dairv2x标签类别首字母大写
        cls_map = {
            'car': 'Car',
            # 'van': 'Van',
            'truck': 'Truck',
            'bus': 'Bus',
            'pedestrian': 'Pedestrian',
            'bicycle': 'Cyclist',
            'motorcycle': 'Motorcyclist'
        }
        cls = cls_map.get(obj['class'], 'unknown')
        x, y, z = obj['x'], obj['y'], obj['z']
        dx, dy, dz = obj['dx'], obj['dy'], obj['dz']
        rx, ry, rz = obj['rx'], obj['ry'], obj['rz']
        dairv2x_obj.update({
            "type": cls,
            "3d_dimensions": {"h": dz, "w": dy, "l": dx},
            "3d_location": {"x": x, "y": y, "z": z},
            "rotation": rz
        })
        dairv2x_label.append(dairv2x_obj)
    return dairv2x_label

def griffin_label_to_dairv2x_label_coop(label):
    dairv2x_label=[]
    for obj in label:
        dairv2x_obj=obj.copy()
        # griffin的标签类别与dairv2x的标签类别不完全一致，这里做一个简单的映射
        # grffin标签类别都是小写，而dairv2x标签类别首字母大写
        cls_map = {
            'car': 'Car',
            # 'van': 'Van',
            'truck': 'Truck',
            'bus': 'Bus',
            'pedestrian': 'Pedestrian',
            'bicycle': 'Cyclist',
            'motorcycle': 'Motorcyclist'
        }
        cls = cls_map.get(obj['class'], 'unknown')
        x, y, z = obj['x'], obj['y'], obj['z']
        dx, dy, dz = obj['dx'], obj['dy'], obj['dz']
        rx, ry, rz = obj['rx'], obj['ry'], obj['rz']



        hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
        # 8 corners around origin
        corners = np.array([
            [ hx,  hy, -hz],
            [ hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx,  hy, -hz],
            [ hx,  hy,  hz],
            [ hx, -hy,  hz],
            [-hx, -hy,  hz],
            [-hx,  hy,  hz],
        ], dtype=np.float32)

        c, s = np.cos(rz), np.sin(rz)
        R = np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        corners = (R @ corners.T).T
        corners += np.array([x, y, z], dtype=np.float32)


        dairv2x_obj.update({
            "type": cls,
            "3d_dimensions": {"h": dz, "w": dy, "l": dx},
            "3d_location": {"x": x, "y": y, "z": z},
            "rotation": rz,
            "world_8_points": corners.tolist()
        })
        dairv2x_label.append(dairv2x_obj)
    return dairv2x_label




# dairv2x位姿是角度，转换griffin位姿同样为角度值
def griffin_pose_to_dairv2x_pose(pose):
    x, y, z, roll, yaw, pitch = pose
    roll = np.rad2deg(roll)
    yaw = np.rad2deg(yaw)
    pitch = np.rad2deg(pitch)
    return [x, y, z, roll, yaw, pitch]

    
# --------------标签坐标系统一与合并方法-----------------------------------------------------
def get_transformation_matrix(pose):
    x, y, z, roll, yaw, pitch = pose
    t=np.array([x, y, z], dtype=float)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R, t

def transform_labels(labels, pose):
    R, t = get_transformation_matrix(pose)
    transformed_labels = []
    for obj in labels:
        center = np.array([obj['x'], obj['y'], obj['z']])
        transformed_center = R @ center + t
        obj_transformed = obj.copy()
        obj_transformed['x'] = transformed_center[0]
        obj_transformed['y'] = transformed_center[1]
        obj_transformed['z'] = transformed_center[2]
        #转换角度
        roll= obj.get('rx', 0.0)
        yaw = obj.get('rz', 0.0)
        pitch = obj.get('ry', 0.0)

        cr,sr= np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)


        Rx = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr, cr]])
        Ry = np.array([[cp, 0, sp],
                       [0, 1, 0],
                       [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0],
                       [sy, cy, 0],
                       [0, 0, 1]])
        R_obj = Rz @ Ry @ Rx
        
        # 变换后的旋转矩阵
        R_new = R @ R_obj
        
        # 从变换后的旋转矩阵提取新的欧拉角
        new_roll = np.arctan2(R_new[2,1], R_new[2,2])
        new_pitch = np.arcsin(-R_new[2,0])
        new_yaw = np.arctan2(R_new[1,0], R_new[0,0])
        
        obj_transformed['rx'] = new_roll
        obj_transformed['ry'] = new_pitch
        obj_transformed['rz'] = new_yaw
        transformed_labels.append(obj_transformed)
    return transformed_labels

#根据重叠度合并标签,重叠度高的使用lidar标签
def merge_labels(labels_lidar, labels_cam, iou_threshold=0.5):
    merged=labels_lidar.copy()  # 先保留所有lidar标签
    
    # merged = []
    # for lidar_obj in labels_lidar:
    #     max_iou = 0
    #     matched_cam_obj = None
    #     for cam_obj in labels_cam:
    #         iou = calculate_iou(lidar_obj, cam_obj)
    #         if iou > max_iou and iou > iou_threshold:
    #             max_iou = iou
    #             matched_cam_obj = cam_obj
    #     if matched_cam_obj is not None:
    #         # 保留lidar标签
    #         merged.append(lidar_obj)
    #     else:
    #         merged.append(lidar_obj)

    # 添加未匹配的cam标签
    for cam_obj in labels_cam:
        is_matched = any(calculate_iou(cam_obj, lidar_obj) > iou_threshold for lidar_obj in labels_lidar)
        if not is_matched:
            merged.append(cam_obj)
    return merged

def calculate_iou(obj1, obj2):
    # 计算两个3D框的IoU，简化为2D BEV IoU
    x1_min = obj1['x'] - obj1['dx']/2
    x1_max = obj1['x'] + obj1['dx']/2
    y1_min = obj1['y'] - obj1['dy']/2
    y1_max = obj1['y'] + obj1['dy']/2

    x2_min = obj2['x'] - obj2['dx']/2
    x2_max = obj2['x'] + obj2['dx']/2
    y2_min = obj2['y'] - obj2['dy']/2
    y2_max = obj2['y'] + obj2['dy']/2

    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (area1 + area2 - inter_area)
    return iou
# ---------------相机标签过滤方法-----------------------------------------------------
def euler_to_R(rx, ry, rz):
    #角度转弧度
    # rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx
# 生成 3D 盒子8个角点（相对于世界坐标系）
def box_corners_3d(obj):
    dx, dy, dz = obj['dx'], obj['dy'], obj['dz']
    # 以物体中心为原点，8个角点（x,y,z）
    hx, hy, hz = dx/2.0, dy/2.0, dz/2.0
    corners = np.array([
        [ hx,  hy,  hz],
        [ hx, -hy,  hz],
        [-hx, -hy,  hz],
        [-hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy, -hz],
        [-hx, -hy, -hz],
        [-hx,  hy, -hz],
    ], dtype=float).T  # 3x8
    R = euler_to_R(obj['rx'], obj['ry'], obj['rz'])
    corners_world = (R @ corners) + np.array([[obj['x']],[obj['y']],[obj['z']]])
    return corners_world.T  # 8x3

# 将 3D 点投影到图像上，返回 Nx2 像素坐标（剔除 Z<=0）
def project_points(pts3d, K, extrinsic):
    pts = np.asarray(pts3d, dtype=float)
    # 构造 Rt (3x4)
    ext = np.asarray(extrinsic, dtype=float)
    if ext.shape == (4,4):
        Rt = ext[:3,:4]
    elif ext.shape == (3,4):
        Rt = ext
    else:
        raise ValueError('extrinsic shape must be 4x4 or 3x4')
    # 变换到相机坐标
    homo = np.concatenate([pts, np.ones((pts.shape[0],1))], axis=1).T  # 4xN
    cam = Rt @ homo  # 3xN
    zs = cam[2,:]
    proj = (K @ cam)  # 3xN
    uv = (proj[:2,:] / proj[2:3,:]).T  # Nx2
    return uv, zs
# 过滤标签：投影后如果所有角点都不在图像范围内或在相机后方，则移除该标签
def filter_labels_in_image(objs, K, extrinsic, img_shape):
    """
    objs: list of object dicts
    K: 3x3 intrinsic
    extrinsic: 3x4 or 4x4
    img_shape: image shape (H, W, [C])
    返回过滤后的 objs，只保留至少有一个角点在图像范围内且在相机前方的目标
    """
    h, w = img_shape[0], img_shape[1]
    kept = []
    for obj in objs:
        corners = box_corners_3d(obj)  # 8x3
        uv, zs = project_points(corners, K, extrinsic)
        zs = np.asarray(zs)
        u = uv[:,0]
        v = uv[:,1]
        # 条件：点在相机前方且像素坐标在 [0,w) x [0,h)
        valid = (zs > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if np.any(valid):
            kept.append(obj)
    return kept

# ========================================
# import copy
# from opencood.utils.box_utils import corner_to_center 
# from opencood.utils.box_utils import project_box3d
# from opencood.utils.box_utils import boxes_to_corners_3d
# from opencood.utils.box_utils import mask_boxes_outside_range_numpy
# def load_single_objects_griffin_hetero(object_list,
#                           output_dict,
#                           lidar_range,
#                           trans_mat,
#                           order):
#     """

#     Parameters
#     ----------
#     object_list : list
#         The list contains all objects surrounding a certain cav.

#     output_dict : dict
#         key: object id, value: object bbx (xyzlwhyaw).

#     lidar_range : list
#          [minx, miny, minz, maxx, maxy, maxz]

#     order : str
#         'lwh' or 'hwl'
#     """

#     i = 0
#     for object_content in object_list:        
#         object_id = i
#         x = object_content['3d_location']['x']
#         y = object_content['3d_location']['y']
#         z = object_content['3d_location']['z']
#         l = object_content['3d_dimensions']['l']
#         h = object_content['3d_dimensions']['h']
#         w = object_content['3d_dimensions']['w']
#         rotation = object_content['rotation']

#         if isinstance(x, str): # in camera label, xyz are str
#             x = eval(x)
#             y = eval(y)
#             z = eval(z)

#         if l==0 or h ==0 or w==0:
#             continue
#         i = i + 1

#         lidar_range_z_larger = copy.deepcopy(lidar_range)
#         lidar_range_z_larger[2] -= 1
#         lidar_range_z_larger[5] += 1

#         bbx_lidar = [x,y,z,h,w,l,rotation] if order=="hwl" else [x,y,z,l,w,h,rotation] # suppose order is in ['hwl', 'lwh']
#         bbx_lidar = np.array(bbx_lidar).reshape(1,-1) # [1,7]
#         bbx_lidar_ego = corner_to_center(
#                             project_box3d(boxes_to_corners_3d(bbx_lidar, order), trans_mat) , order=order)
#         bbx_lidar_ego = mask_boxes_outside_range_numpy(bbx_lidar_ego, lidar_range_z_larger, order)

#         if bbx_lidar_ego.shape[0] > 0:
#             if object_content['type'] == "Car" or \
#                object_content['type'] == "Van" or \
#                object_content['type'] == "Truck" or \
#                object_content['type'] == "Bus":
#                     output_dict.update({object_id: bbx_lidar_ego})

# pose=read_pose_json_griffin(r"D:\Zgapyear\UGV-UAV\DAIRV2X\GRIFFIN\drone_metadata\griffin_50scenes_25m\griffin-release\drone-side\pose\011185.json")
# print(type(pose[0]))