# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 12:16
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : sort_val_result.py
# @Software: PyCharm

import os
import glob
import argparse
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file

def find_result_tb_file(model_path: str, eval=False):
    if not os.path.exists(model_path):
        raise FileNotFoundError("model path not found")
    if not eval:
        tb_path = os.path.join(model_path, "tensorboard")
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
    # set default config of pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="", required=True)
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    args = parser.parse_args()

    # make the root dir of model path
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    cfg.TAG = Path(args.cfg_file).stem
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    print(output_dir)
    tb_file_list = find_result_tb_file(str(output_dir), eval=True)


    print("[Hard]")
    # find the best performance of model epoch
    aim_tags = ["Car_3d/hard_R40", "Pedestrian_3d/hard_R40", "Cyclist_3d/hard_R40"]
    for tb_file in tb_file_list:
        try:
                print("===========================================================")
                tb_data_df = read_tb_file(tb_file)
                print("Total records: {}".format(len(tb_data_df)))
                tb_data_df['hard_R40'] = tb_data_df[aim_tags].mean(axis=1)
                data_df_t3 = tb_data_df.sort_values(by="hard_R40", ascending=False).head(3)
                data_df_t3 = data_df_t3[["step", "hard_R40", *aim_tags]]
                print(data_df_t3)
                print("===========================================================")
        except Exception as e:
            print("Error:", e)
            print("Jump the file:", tb_file)


    print("[Moderate]")
    # find the best performance of model epoch
    aim_tags = ["Car_3d/moderate_R40", "Pedestrian_3d/moderate_R40", "Cyclist_3d/moderate_R40"]
    for tb_file in tb_file_list:
        try:
                print("===========================================================")
                tb_data_df = read_tb_file(tb_file)
                print("Total records: {}".format(len(tb_data_df)))
                tb_data_df['moderate_R40'] = tb_data_df[aim_tags].mean(axis=1)
                data_df_t3 = tb_data_df.sort_values(by="moderate_R40", ascending=False).head(3)
                data_df_t3 = data_df_t3[["step", "moderate_R40", *aim_tags]]
                print(data_df_t3)
                print("===========================================================")
        except Exception as e:
            print("Error:", e)
            print("Jump the file:", tb_file)