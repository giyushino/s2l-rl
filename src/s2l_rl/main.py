#conda_env: s2l
import os
#import torch
#import multiprocessing as mp
#from datasets import load_dataset

#from vllm import LLM, SamplingParams
#from vllm.lora.request import LoRARequest
CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIRECTORY = os.path.dirname(CURRENT_FILE_PATH)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIRECTORY))

def main(old_policy_path, current_policy_path, num_gpus, start_gpu):
    for i in range(num_gpus):
        gpu_id = i + start_gpu
        print(f"{gpu_id}")

    return
    



if __name__ == "__main__":
    CURRENT_FILE_PATH = os.path.abspath(__file__)
    CURRENT_DIRECTORY = os.path.dirname(CURRENT_FILE_PATH)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIRECTORY))
    main(None, None, 4, 4)
    #load_dataset("parquet", data_files="file.parquet")


#    lora_requests = {
#        old_policy: LoRARequest("old_policy", 1, old_policy_path),
#        current_policy: LoRARequest("current_policy", 2, new_policy_path),
#    }
