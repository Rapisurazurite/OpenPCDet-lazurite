# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 12:16
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : sort_val_result.py
# @Software: PyCharm

import os
import glob
import pickle


def find_result_pickle_file(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError("model path not found")

    pass


def read_pickle_file(file_path: str):
    info = pickle.load(open(file_path, "rb"))
    return info


def main():
    pass


if __name__ == '__main__':
    file_path = "/mnt/d/Storage/Download/Compressed/result.pkl"
    read_pickle_file(file_path)
