import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from functools import partial
from torch.utils.data import Dataset
from PIL import Image
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.camera_utils import load_camera_data, load_intrinsic_DAIR_V2X
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose, rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets.basedataset.dataset_untils import griffin_untils
# from dataset_untils import griffin_untils

class GRIFFINBaseDataset(Dataset):
    def __init__(self, params, visualize=False, train=True):
        self.params = params
        self.visualize = visualize
        self.train=train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou

        if 'data_augment' in params: # late and early
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else: # intermediate
            self.data_augmentor = None

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        assert self.load_depth_file is False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                                    else self.generate_object_center_camera

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']

        self.split_info = read_json(split_dir)
        # co_datainfo = read_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        # self.co_data = OrderedDict()
        # for frame_info in co_datainfo:
        #     veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
        #     self.co_data[veh_frame_id] = frame_info

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        

    def reinitialize(self):
        pass

    """
    griffin数据格式-
    data/GRIFFIN
      |-drone_camera_bottom\griffin_50scenes_25m\griffin-release\drone-side\camera\bottom 无人机摄像头数据.png
      |-drone_metadata\griffin_50scenes_25m\griffin-release\drone-side 无人机元数据
        |-calib 相机定标
        |-label 标签
        |-pose  位姿
      |-vehicle_lidar\griffin_50scenes_25m\griffin-release\vehicle-side\lidar\lidar_top  车载激光雷达数据.ply
      |-vehicle_metadata\griffin_50scenes_25m\griffin-release\vehicle-side 车载元数据
        |-calib 相机定标
        |-label 标签
        |-pose  位姿
      |-train.json 训练集
      |-val.json   验证集
    """

    def retrieve_base_data(self, idx):
        data=OrderedDict()
        veh_frame_id=self.split_info[idx]
        veh_lidar_path=os.path.join(self.root_dir, 'vehicle_lidar/griffin_50scenes_25m/griffin-release/vehicle-side/lidar/lidar_top', veh_frame_id+'.ply')
        veh_pose_path=os.path.join(self.root_dir, 'vehicle_metadata/griffin_50scenes_25m/griffin-release/vehicle-side/pose', veh_frame_id+'.json')
        veh_label_path=os.path.join(self.root_dir, 'vehicle_metadata/griffin_50scenes_25m/griffin-release/vehicle-side/label', veh_frame_id+'.txt')

        drone_camera_path=os.path.join(self.root_dir, 'drone_camera_bottom/griffin_50scenes_25m/griffin-release/drone-side/camera/bottom', veh_frame_id+'.png')
        drone_pose_path=os.path.join(self.root_dir, 'drone_metadata/griffin_50scenes_25m/griffin-release/drone-side/pose', veh_frame_id+'.json')
        drone_label_path=os.path.join(self.root_dir, 'drone_metadata/griffin_50scenes_25m/griffin-release/drone-side/label', veh_frame_id+'.txt')
        drone_calib_path=os.path.join(self.root_dir, 'drone_metadata/griffin_50scenes_25m/griffin-release/drone-side/calib', 'bottom.json')

        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()
        intrinsic, extrinsic, distortion = griffin_untils.load_calibration_json_griffin(drone_calib_path)

        data[0]['params']['lidar_pose']= griffin_untils.read_pose_json_griffin(veh_pose_path)
        data[1]['params']['lidar_pose']= griffin_untils.read_pose_json_griffin(drone_pose_path)

        data[0]['params']['vehicles_single_all'] = griffin_untils.load_label_txt(veh_label_path)
        #griffin是五个镜头的标签，这里过滤标签只保留无人机底部摄像头的标签
        data[1]['params']['vehicles_single_all'] = griffin_untils.filter_labels_in_image(griffin_untils.load_label_txt(drone_label_path), intrinsic, extrinsic, (1080, 1920))
        
        #合并标签
        data[0]['params']['vehicles_all'] =\
              griffin_untils.merge_labels(
                  griffin_untils.transform_labels(data[0]['params']['vehicles_single_all'], data[0]['params']['lidar_pose']), 
                  griffin_untils.transform_labels(data[1]['params']['vehicles_single_all'], data[1]['params']['lidar_pose']),
                  iou_threshold=0.8)
        data[1]['params']['vehicles_all']=[]
     
        if self.load_camera_file:
            data[0]['camera_data']=[]
            data[0]['params']['camera0'] = OrderedDict()
            data[0][ 'params']['camera0']['extrinsic'] = []
            data[0]['params']['camera0']['intrinsic'] = []

            data[1]['camera_data']=griffin_untils.load_camera_data_griffin([drone_camera_path])
            data[1]['params']['camera0'] = OrderedDict()

            data[1]['params']['camera0']['extrinsic'] = extrinsic #这里的外参是相机到无人机坐标系的平移旋转4*4矩阵
            data[1]['params']['camera0']['intrinsic'] = intrinsic

        if self.load_lidar_file or self.visualize:
            data[0]['lidar_np']=griffin_untils.load_lidar_ply_griffin(veh_lidar_path)
            # data[1]['lidar_np']=[]
            data[1]['lidar_np']=np.zeros((0,4), dtype=np.float32) # 无人机没有激光雷达数据，设置为空

        

        #将所有标签转为dairv2x格式
        data[0]['params']['vehicles_all'] = griffin_untils.griffin_label_to_dairv2x_label_coop(data[0]['params']['vehicles_all'])
        data[0]['params']['vehicles_front'] = data[0]['params']['vehicles_all']
        data[1]['params']['vehicles_all'] = []
        data[1]['params']['vehicles_front'] = []

        data[0]['params']['vehicles_single_all'] = griffin_untils.griffin_label_to_dairv2x_label(data[0]['params']['vehicles_single_all'])
        data[1]['params']['vehicles_single_all'] = griffin_untils.griffin_label_to_dairv2x_label(data[1]['params']['vehicles_single_all'])

        data[0]['params']['vehicles_single_front'] = data[0]['params']['vehicles_single_all']
        data[1]['params']['vehicles_single_front'] = data[1]['params']['vehicles_single_all']


        # pose转换为dairv2x格式
        data[0]['params']['lidar_pose'] = griffin_untils.griffin_pose_to_dairv2x_pose(data[0]['params']['lidar_pose'])
        data[1]['params']['lidar_pose'] = griffin_untils.griffin_pose_to_dairv2x_pose(data[1]['params']['lidar_pose'])

        if getattr(self, "heterogeneous", False):
            self.generate_object_center_lidar = \
                                partial(self.generate_object_center_single_hetero, modality='lidar')
            self.generate_object_center_camera = \
                                partial(self.generate_object_center_single_hetero, modality='camera')

            # by default
            data[0]['modality_name'] = 'm1'
            data[1]['modality_name'] = 'm2'
            # veh cam inf lidar. We don't need json to assign the initial modality
            # data[0]['modality_name'] = 'm2'
            # data[1]['modality_name'] = 'm1'

            # if self.train: # randomly choose RSU or Veh to be Ego
            #     p = np.random.rand()
            #     # if p > 0.5:
            #     #     data[0], data[1] = data[0], data[1]
            #     #     data[0]['ego'] = True
            #     #     data[1]['ego'] = False
            # else:
            #     # evaluate, the agent of ego modality should be ego
            #     if self.adaptor.mapping_dict[data[0]['modality_name']] not in self.ego_modality and \
            #         self.adaptor.mapping_dict[data[1]['modality_name']] in self.ego_modality:
            #         data[0], data[1] = data[0], data[1]
            #         # data[0]['ego'] = True
            #         # data[1]['ego'] = False

            # data[0]['modality_name'] = self.adaptor.reassign_cav_modality(data[0]['modality_name'], 0)
            # data[1]['modality_name'] = self.adaptor.reassign_cav_modality(data[1]['modality_name'], 1)



        return data

    def __len__(self):
        return len(self.split_info)
    
    def __getitem__(self, idx):
        pass

    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_all']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_front']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
                                                        
    ### Add new func for single side
    def generate_object_center_single(self,
                               cav_contents,
                               reference_lidar_pose,
                               **kwargs):
        """
        veh or inf 's coordinate. 

        reference_lidar_pose is of no use.
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if self.label_type == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)

    ### Add for heterogeneous, transforming the single label from self coord. to ego coord.
    def generate_object_center_single_hetero(self,
                                            cav_contents,
                                            reference_lidar_pose, 
                                            modality):
        """
        loading the object from single agent. 
        
        The same as *generate_object_center_single*, but it will transform the object to reference(ego) coordinate,
        using reference_lidar_pose.
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if modality == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single_hetero(cav_contents, reference_lidar_pose, suffix)


    def get_ext_int(self, params, camera_id):
        lidar_to_camera = params["camera%d" % camera_id]['extrinsic'].astype(np.float32) # R_cw
        camera_to_lidar = np.linalg.inv(lidar_to_camera) # R_wc
        camera_intrinsic = params["camera%d" % camera_id]['intrinsic'].astype(np.float32
        )
        return camera_to_lidar, camera_intrinsic


    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.
        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape
        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw
        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask



    
