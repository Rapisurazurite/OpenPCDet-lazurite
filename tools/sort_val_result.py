# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 12:16
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : sort_val_result.py
# @Software: PyCharm

import os
import glob
import pickle
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


def find_result_tb_file(model_path: str, eval=False):
    if not os.path.exists(model_path):
        raise FileNotFoundError("model path not found")
    if not eval:
        tb_path = os.path.join(model_path, "default", "tensorboard")
    else:
        tb_path = os.path.join(model_path, "eval", "eval_all_default", "default", "tensorboard_val")
    tb_path_list = glob.glob(os.path.join(tb_path, r"events.out.tfevents.*"))
    return tb_path_list


def read_tb_file(file_path: str):
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()
    keys = ea.Tags()["scalars"]

    data_df = pd.DataFrame(columns=["step", *keys])
    for i in range(len(ea.Scalars(keys[0]))):
        step = ea.Scalars(keys[0])[i].step
        data_row = [step, *[ea.Scalars(key)[i].value for key in keys]]
        data_df.loc[i] = data_row
    return data_df


def main():
    # centerpoint
    file_path = "/mnt/d/Storage/Download/Compressed/events.out.tfevents.1646983831.512d887dcf72"
    # centerpoint_dyn
    file_path = "/mnt/d/Storage/Download/Compressed/events.out.tfevents.1646799743.de1a1ae0f9f8"

    data_df = read_tb_file(file_path)
    aim_tags = ["Car_3d/hard_R40", "Pedestrian_3d/hard_R40", "Cyclist_3d/hard_R40"]
    data_df['hard_R40'] = data_df[aim_tags].mean(axis=1)
    # select the top 3 rows
    data_df_t3 = data_df.sort_values(by="hard_R40", ascending=False).head(3)
    data_df_t3 = data_df_t3[["step", "hard_R40", *aim_tags]]
    print(data_df_t3)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    main()
