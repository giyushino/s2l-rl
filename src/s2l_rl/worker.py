#conda_env: s2l
import os
#import sys
import json
#import subprocess
#import torch
import pandas as pd

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from s2l_rl.grader import extract_solution, normalize_latex_string

def start_engine(model_name: str, policy_path: str, old_policy: bool):
    if old_policy:
        lora_req = LoRARequest("old_policy", 1, policy_path)
    else:
        lora_req = LoRARequest("current_policy", 1, policy_path)

    llm = LLM(model=model_name, enable_lora=True, max_lora_rank=64)
    return llm, lora_req

# to do: change dataset loading, we can do that later
def main(model_name: str, policy_path: str, gpu_id: int,  dataset_path: str, old_policy: bool, start_id: int, dataset_length: int, rollout_size: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # load the model and lora adapters
    llm, lora_req = start_engine(model_name, policy_path, old_policy)
    
    df = pd.read_parquet(dataset_path)
    prompts = []
    ground_truths = []
    for idx, row in df.iterrows():
#        if idx >= start_id and idx < start_id + dataset_length:
        prompts.append(row["prompt"][0]["content"])
        ground_truths.append(row["reward_model"]["ground_truth"])
    
    import time
    t0 = time.perf_counter()
    outputs = llm.generate(
        prompts[:1],
        SamplingParams(max_tokens=10, logprobs=1, prompt_logprobs=1),
        lora_request=lora_req,
    )
    t1 = time.perf_counter()

    print(f"time taken: {t1 - t0}")
    test = outputs[0]
    print(dir(test))
    
    



#    with open("/home/allanz/s2l-rl/test.json", "w") as f:
#        json.dump(outputs[0], f)

if __name__ == "__main__":
    main("Qwen/Qwen2.5-1.5B-Instruct", "/home/allanz/s2l-rl/checkpoints/global_step_10/actor/lora_adapter/", 0, "/home/allanz/s2l-rl/datasets/math12k/train.parquet", False, None, None, None)
