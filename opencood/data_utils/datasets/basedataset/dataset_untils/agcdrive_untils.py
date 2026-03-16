import open3d as o3d
import numpy as np
import yaml
def load_lidar_pcd(file_path):
    """_summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """    
    pcd = o3d.t.io.read_point_cloud(file_path)
    lidarpoints = pcd.point['positions'].numpy()  
    # 第四维度是强度信息，agcdrive缺失，当作0
    intensity = np.zeros((lidarpoints.shape[0], 1))
    # intensity = pcd.point['I'].numpy() 
    # distances = np.linalg.norm(lidarpoints, axis=1)
    # threshold = 100.0  # 设置距离阈值
    # filtered_points = lidarpoints[distances < threshold]
    data=np.hstack((lidarpoints, intensity))  # 将强度信息添加到点云数据中
    return data

def load_json(file_path):
    """_summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """    
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_yaml(file_path):
    """_summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """    
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def load_pose_yaml_agcdrive(file_path):
    """_summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """    
    data = load_yaml(file_path)
    pose = data['lidar_pose']
    return pose

def load_pairwise_t_matrix_json_agcdrive(file_path):
    """_summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """    
    data = load_json(file_path)
    pairwise_t_matrix1 = np.asanyarray(data[0]['pairwise_t_matrix1'])
    pairwise_t_matrix2 = np.asanyarray(data[1]['pairwise_t_matrix2'])
    return pairwise_t_matrix1, pairwise_t_matrix2

def pairwise_t_matrix_to_pose(matrix):
    """_summary_

    Args:
        pairwise_t_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """    
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    pitch = np.arctan2(-matrix[2, 0], np.sqrt(matrix[0, 0]**2 + matrix[1, 0]**2))
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    x=matrix[0, 3]
    y=matrix[1, 3]
    z=matrix[2, 3]
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    return [x, y, z, roll, yaw, pitch]
    # 提取旋转矩阵和平移向量
    # rotation_matrix = pairwise_t_matrix[:3, :3]
    # translation_vector = pairwise_t_matrix[:3, 3]

    # # 将旋转矩阵转换为欧拉角（roll, pitch, yaw）
    # roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    # pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2))
    # yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # return translation_vector, (roll, pitch, yaw)

def load_label_json_agcdrive(file_path):
    """_summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """    
    data = load_json(file_path)
    label= data[2]['objects']
    return label

def agcdrive_to_dairv2x_label(label):
    """_summary_

    Args:
        label (_type_): _description_

    Returns:
        _type_: _description_
    """    
    dairv2x_label=[]
    for obj in label:
        dairv2x_obj=obj.copy()
        name=obj.get('className', 'unknown')

        contour = obj.get("contour", {})
        center3d = contour.get("center3D")
        size3d = contour.get("size3D")
        rot3d = contour.get("rotation3D")

        x,y,z=center3d["x"], center3d["y"], center3d["z"]
        dx,dy,dz=size3d["x"], size3d["y"], size3d["z"]
        rx, ry, rz=rot3d["x"], rot3d["y"], rot3d["z"]

        dairv2x_obj.update({
            "type": name,
            "3d_dimensions": {"h": dz, "w": dy, "l": dx},
            "3d_location": {"x": x, "y": y, "z": z},
            "rotation": rz
        })
        dairv2x_label.append(dairv2x_obj)
    return dairv2x_label