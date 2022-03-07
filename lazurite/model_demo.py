# -*- coding: utf-8 -*-
# @Time    : 2022/2/26 14:36
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : model_demo.py
# @Software: PyCharm
import glob
from pathlib import Path
import numpy as np
import torch

from pcdet.utils import common_utils
from pcdet.config import cfg_from_yaml_file, cfg
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
import sys, os
import open3d
from tools.visual_utils import open3d_vis_utils as V

# current dir
os.chdir("/home/lazurite/code/OpenPCDet/tools")
# %%
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


#%%
logger = common_utils.create_logger()
# data_path = "../data/kitti/training/velodyne"
# model_cfg_file = "cfgs/kitti_models/pointpillar.yaml"
# model_ckpt_file = "ckpt/pointpillar_7728.pth"
data_path = "../data/nuscenes/v1.0-trainval/samples/LIDAR_TOP"
# model_cfg_file = "cfgs/kitti_models/centerPoint.yaml"
# model_ckpt_file = "ckpt/centerpoint_checkpoint_epoch_80.pth"

model_cfg_file = "cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml"
model_ckpt_file = "ckpt/cbgs_voxel01_centerpoint_nds_6454.pth"
cfg_from_yaml_file(model_cfg_file, cfg)

#%%
demo_dataset = DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    root_path=Path(data_path), ext=".bin", logger=logger
)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
model.load_params_from_file(filename=model_ckpt_file, logger=logger, to_cpu=True)
#%%


#%%
model.cuda()
model.eval()

with torch.no_grad():
    for idx, data_dict in enumerate(demo_dataset):
        logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)
        V.draw_scenes(
            points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        )