# -*- coding: utf-8 -*-
# @Time    : 4/3/2022 6:28 PM
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : h5_create.py
# @Software: PyCharm
import os
import tqdm
import h5py
import numpy as np

ROOT_PATH = "../data/nuscenes/v1.0-trainval/sweeps/LIDAR_TOP/"

f = h5py.File("../data/nuscenes/v1.0-trainval/sweeps.h5", "w")
example_data = np.fromfile(ROOT_PATH + "n015-2018-07-18-11-50-34+0800__LIDAR_TOP__1531886336148818.pcd.bin", dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
bin_data = os.listdir(ROOT_PATH)
n_bin_data = len(bin_data)

print("Creating h5 file...")
print("Number of bins:", n_bin_data)

np_dt = h5py.special_dtype(vlen=np.dtype('float32'))
dset = f.create_dataset("sweeps_data", shape=(n_bin_data, ), dtype=np_dt)
str_dt = h5py.special_dtype(vlen=str)
name_map = f.create_dataset("sweeps_name", (n_bin_data), dtype=str_dt)

pbar = tqdm.tqdm(total=n_bin_data)
for i, bin_file in enumerate(bin_data):
    dset[i] = np.fromfile(ROOT_PATH + bin_file, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4].flatten()
    name_map[i] = bin_file
    pbar.update(1)

f.close()
print("Done!")
