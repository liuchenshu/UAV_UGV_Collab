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
from opencood.data_utils.datasets.basedataset.dataset_untils import agcdrive_untils
# from dataset_untils import griffin_untils

class AGCDRIVEBaseDataset(Dataset):
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
    AGCDRIVE数据格式
    data/AGCDRIVE/AGCDRIVE
        ├── 1  .pcd点云
        |—— 2  .pcd点云
        |—— 3  .pcd点云
        |—— cooperative/ .json文件，包含每一帧的标签和pose等信息
        train.json
        val.json
    """

    def retrieve_base_data(self, idx):
        data=OrderedDict()
        veh_frame_id=self.split_info[idx]

        lidar_agent1_path=os.path.join(self.root_dir, '1', veh_frame_id+'.pcd')
        lidar_agent2_path=os.path.join(self.root_dir, '2', veh_frame_id+'.pcd')
        lidar_agent3_path=os.path.join(self.root_dir, '3', veh_frame_id+'.pcd')
        cooperative_path=os.path.join(self.root_dir, 'cooperative', veh_frame_id+'.json')

        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False
        data[2] = OrderedDict()
        data[2]['ego']=False

        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()
        data[2]['params'] = OrderedDict()

        pair_t_matrix1, pair_t_matrix2 = agcdrive_untils.load_pairwise_t_matrix_json_agcdrive(cooperative_path)
        pose1 = agcdrive_untils.pairwise_t_matrix_to_pose(pair_t_matrix1)
        pose2 = agcdrive_untils.pairwise_t_matrix_to_pose(pair_t_matrix2)

        data[0]['params']['lidar_pose'] = [0, 0, 0, 0, 0, 0] # ego pose is always [0, 0, 0, 0, 0, 0]
        data[1]['params']['lidar_pose'] = pose1
        data[2]['params']['lidar_pose'] = pose2

        data[0]['params']['vehicles_all'] = agcdrive_untils.load_label_json_agcdrive(cooperative_path)
        data[0]['params']['vehicles_front'] = data[0]['params']['vehicles_all']
        data[1]['params']['vehicles_all'] = []
        data[1]['params']['vehicles_front'] = []
        data[2]['params']['vehicles_all'] = []
        data[2]['params']['vehicles_front'] = []

        data[0]['params']['vehicles_single_all']=agcdrive_untils.agcdrive_to_dairv2x_label(data[0]['params']['vehicles_all'])
        data[1]['params']['vehicles_single_all']=[]
        data[2]['params']['vehicles_single_all']=[]

        data[0]['params']['vehicles_single_front']=data[0]['params']['vehicles_single_all']
        data[1]['params']['vehicles_single_front']=[]
        data[2]['params']['vehicles_single_front']=[]

        if self.load_lidar_file or self.visualize:
            data[0]['lidar_np']=agcdrive_untils.load_lidar_pcd(lidar_agent1_path)
            data[1]['lidar_np']=agcdrive_untils.load_lidar_pcd(lidar_agent2_path)
            data[2]['lidar_np']=agcdrive_untils.load_lidar_pcd(lidar_agent3_path)

        if getattr(self, "heterogeneous", False):
            self.generate_object_center_lidar = \
                                partial(self.generate_object_center_single_hetero, modality='lidar')
            self.generate_object_center_camera = \
                                partial(self.generate_object_center_single_hetero, modality='camera')

            # by default
            data[0]['modality_name'] = 'm1'
            data[1]['modality_name'] = 'm2'
            data[2]['modality_name'] = 'm3'


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



    
