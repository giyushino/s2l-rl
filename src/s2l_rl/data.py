#conda_env: comp
import re
import ast
import json
from datasets import load_dataset, Dataset
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_directory)) #/home/user/s2l-rl

def load_math_12k(mode: str="train", data_path=None):
    if mode == "train":
        return load_dataset("parquet", data_files="/home/allanz/s2l-rl/datasets/math12k/train.parquet")
    elif mode == "eval":
        return load_dataset("parquet", data_files="/home/allanz/s2l-rl/datasets/math12k/train.parquet")

if __name__ == "__main__":
    dataset = load_dataset("hiyouga/math12k")
    print(dataset)
    
