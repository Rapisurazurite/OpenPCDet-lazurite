# -*- coding: utf-8 -*-
# @Time    : 2022/2/26 14:25
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : data_demo.py
# @Software: PyCharm
import os

from pcdet.utils import common_utils
from pcdet.config import cfg_from_yaml_file, cfg
from pcdet.datasets import build_dataloader
from tools.visual_utils import open3d_vis_utils as V

# current dir
os.chdir("/home/lazurite/code/OpenPCDet/tools")
# %%

kitti_datset_cfg_path = "tools/cfgs/dataset_configs/kitti_dataset.yaml"
model_cfg_file = "cfgs/kitti_models/pointpillar.yaml"
cfg_from_yaml_file(model_cfg_file, cfg)

# %%
logger = common_utils.create_logger(log_file=None, rank=cfg.LOCAL_RANK)
train_set, train_loader, train_sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=None, workers=4,
    logger=logger,
    training=True,
    merge_all_iters_to_one_epoch=False,
    total_epochs=None
)

# the key of the dict are
# ['frame_id', 'calib', 'gt_boxes', 'road_plane', 'points', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'image_shape']


# %%
# visualize the dataset

index = 0
sample_idx = train_set.kitti_infos[index]['point_cloud']['lidar_idx']
print(sample_idx)
labels = train_set.kitti_infos[index]['annos']['gt_boxes_lidar']
data_dict = train_set[index]
pc_data = train_set.get_lidar(sample_idx)

#%%
V.draw_scenes(
    points=pc_data[:, :-1], ref_boxes=labels,
    ref_scores=None, ref_labels=None
)
