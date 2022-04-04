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

h5_paths = ["../data/nuscenes/v1.0-trainval/samples.h5", "../data/nuscenes/v1.0-trainval/sweeps.h5"]
h5_files = [h5py.File(path, "r") for path in h5_paths]

h5_merge = h5py.File("../data/nuscenes/v1.0-trainval/samples_sweeps.h5", "w")
n_bin_data = h5_files[0]["samples_data"].shape[0] + h5_files[1]["sweeps_data"].shape[0]

print("Creating h5 file...")
print("Number of bins:", n_bin_data)


np_dt = h5py.special_dtype(vlen=np.dtype('float32'))
dset = h5_merge.create_dataset("data", shape=(n_bin_data, ), dtype=np_dt)
str_dt = h5py.special_dtype(vlen=str)
name_map = h5_merge.create_dataset("name", (n_bin_data), dtype=str_dt)


pbar = tqdm.tqdm(total=n_bin_data)
len_samples = h5_files[0]["samples_data"].shape[0]
for i in range(len_samples):
    dset[i] = h5_files[0]["samples_data"][i]
    name_map[i] = h5_files[0]["samples_name"][i]
    pbar.update(1)
for i in range(len_samples, n_bin_data):
    dset[i] = h5_files[1]["sweeps_data"][i - len_samples]
    name_map[i] = h5_files[1]["sweeps_name"][i - len_samples]
    pbar.update(1)
h5_merge.close()
print("Done!")
