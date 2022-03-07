# -*- coding: utf-8 -*-
# @Time    : 2022/2/26 14:25
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : data_demo.py
# @Software: PyCharm
import os

import numpy as np

from pcdet.utils import common_utils
from pcdet.config import cfg_from_yaml_file, cfg
from pcdet.datasets import build_dataloader
from tools.visual_utils import open3d_vis_utils as V

# current dir
os.chdir("/home/lazurite/code/OpenPCDet/tools")
# %%

# model_cfg_file = "cfgs/kitti_models/pointpillar.yaml"
model_cfg_file = "cfgs/nuscenes_models/cbgs_pp_multihead.yaml"
cfg_from_yaml_file(model_cfg_file, cfg)
cfg.DATA_CONFIG.VERSION = "v1.0-mini"
class_names = cfg.CLASS_NAMES
class_names_to_id = {class_names[i]: i for i in range(len(class_names))}
id_to_class_names = {i: class_names[i] for i in range(len(class_names))}
# %%
logger = common_utils.create_logger(log_file=None, rank=cfg.LOCAL_RANK)
train_set, train_loader, train_sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=None, workers=4,
    logger=logger,
    training=False,
    merge_all_iters_to_one_epoch=False,
    total_epochs=None
)

# the key of the dict are
# ['frame_id', 'calib', 'gt_boxes', 'road_plane', 'points', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'image_shape']


# %%
# visualize the dataset

index = 12
pc_data = train_set.get_lidar_with_sweeps(index, max_sweeps=10)
class_data = train_set.infos[index]['gt_names']
ref_boxes = train_set.infos[index]['gt_boxes']


assert len(ref_boxes) == len(class_data)
fliter_id = []
fliter_boxes = []
for i in range(len(ref_boxes)):
    if class_data[i] == 'ignore':
        continue
    fliter_boxes.append(ref_boxes[i])
    fliter_id.append(class_names_to_id[class_data[i]])
fliter_boxes = np.array(fliter_boxes)
fliter_id = np.array(fliter_id)

#%%
V.draw_scenes(
    points=pc_data[:, :-2], ref_boxes=fliter_boxes,
    ref_scores=None, ref_labels=fliter_id
)
