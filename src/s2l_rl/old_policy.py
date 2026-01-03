#conda_env: s2l
import os
import time
import json
import argparse
import pandas as pd

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.inputs import TokensPrompt
#
#from s2l_rl.grader import extract_solution, normalize_latex_string

def start_engine(model_name: str, policy_path: str, gpu_memory_utilization: float = 0.8):
    llm = LLM(model=model_name, enable_lora=True, max_lora_rank=64, enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization)
    if policy_path == "None" or policy_path == "":
        return llm, None
    lora_req = LoRARequest("current_policy", 1, policy_path)
    return llm, lora_req

def main(
    model_name: str,
    policy_path: str,
    gpu_id: int,
    data_path: str,
    save_path: str,
    batch_size: int,
    gpu_memory_utilization: float = 0.8,
    verbose: bool = False
):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # load the model and lora adapters
    llm, lora_req = start_engine(model_name, policy_path, gpu_memory_utilization)
    data = []
    # load the completions from current policy
    with open(data_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    t0 = time.perf_counter()
    print("beginning gen")
    with open(save_path, "w") as file:
        for i in range(0, len(data), batch_size):
            original_data = [data[x] for x in range(i, min(i+batch_size, len(data)))]
            original_prompt_lens = [len(data[x]["prompt_ids"]) for x in range(i, min(i+batch_size, len(data)))]
            batch_prompts = [TokensPrompt(prompt_token_ids=data[x]["prompt_ids"]+data[x]["output_token_ids"]) for x in range(i, min(i+batch_size, len(data)))]

            outputs = llm.generate(
                        batch_prompts,
                        SamplingParams(max_tokens=1, prompt_logprobs=1, n=1),
                        lora_request=lora_req,
                        use_tqdm=False,
                    )

            for output, prompt_length, original in zip(outputs, original_prompt_lens, original_data):
                old_logprobs = []
                # only include the tokens past the old prompt
                for index, (token, prompt_logprob) in enumerate(zip(output.prompt_token_ids, output.prompt_logprobs)):
                    if index >= prompt_length:
                        old_logprobs.append(prompt_logprob[token].logprob)
                original["old_logprobs"] = old_logprobs
                file.write(json.dumps(original))
                file.write("\n")

            if verbose:
                t1 = time.perf_counter()
                elapsed = t1 - t0
                batches_completed = (i // batch_size) + 1
                total_batches = (len(data) + batch_size - 1) // batch_size
                batches_remaining = total_batches - batches_completed
                time_per_batch = elapsed / batches_completed
                eta_seconds = time_per_batch * batches_remaining
                eta_minutes = eta_seconds / 60
                print(f"Batch {batches_completed}/{total_batches} | Elapsed: {elapsed:.2f}s | ETA: {eta_minutes:.2f}min")




    return

def parse_args():
    parser = argparse.ArgumentParser(description="Compute old policy log probabilities for generated sequences")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to old policy LoRA adapter")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save output JSONL file")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for processing")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization (default: 0.8)")
    parser.add_argument("--verbose", action="store_true", help="Print progress information (recommended for GPU 0 only)")
    return parser.parse_args()

if __name__ == "__main__":
    os.environ['TQDM_DISABLE'] = '1'
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    main(
        model_name=args.model_name,
        policy_path=args.policy_path,
        gpu_id=args.gpu_id,
        data_path=args.data_path,
        save_path=args.save_path,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=args.verbose,
    )
