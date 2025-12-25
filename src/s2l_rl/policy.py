#conda_env: s2l
import os
import time
import json
import argparse
import pandas as pd

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from s2l_rl.grader import extract_solution, normalize_latex_string

"""
Goal of this file is to generate sequences + obtain log probs for every element
in the dataset. Will use an offline vLLM engine
"""

def start_engine(model_name: str, policy_path: str, old_policy: bool):
    if old_policy:
        lora_req = LoRARequest("old_policy", 1, policy_path)
    else:
        lora_req = LoRARequest("current_policy", 1, policy_path)

    llm = LLM(model=model_name, enable_lora=True, max_lora_rank=64, enable_prefix_caching=True)
    return llm, lora_req

# to do: change dataset loading, we can do that later
def main(
    model_name: str,
    policy_path: str,
    gpu_id: int,
    dataset_path: str,
    old_policy: bool,
    start_id: int,
    num_samples: int,
    max_tokens: int,
    rollout_size: int,
    batch_size: int,
    save_path: str,
):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # load the model and lora adapters
    llm, lora_req = start_engine(model_name, policy_path, old_policy)
     
    df = pd.read_parquet(dataset_path)
    prompts = []
    ground_truths = []
    
    # this is also dependnt on the dataset, so we should move this out
    for row in df.iloc[start_id:start_id+num_samples].itertuples():
        prompts.append(row[4][0]["content"])
        ground_truths.append(row[6]["ground_truth"])
    t0 = time.perf_counter()
    with open(save_path, "w") as file:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_ground_truths = ground_truths[i:i+batch_size]

            outputs = llm.generate(
                batch_prompts,
                SamplingParams(max_tokens=max_tokens, logprobs=1, n=rollout_size),
                lora_request=lora_req,
            )

            for output, ground_truth in zip(outputs, batch_ground_truths):
                # Process all rollout_size completions for this prompt
                for completion in output.outputs:
                    logprobs = completion.logprobs
                    token_ids = [next(iter(d)) for d in logprobs]
                    log_probs = [next(iter(d.values())).logprob for d in logprobs]
                    model_output = "".join([next(iter(d.values())).decoded_token for d in logprobs])

                    data = {
                        "prompt": output.prompt, "prompt_ids": output.prompt_token_ids, "log_probs": log_probs,
                        "output_token_ids": token_ids, "ground_truth": ground_truth, "model_output": model_output,
                        "model_answer": normalize_latex_string(extract_solution(model_output))
                    }
                    file.write(json.dumps(data))
                    file.write("\n")

            t1 = time.perf_counter()
            print(f"Processed batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}, elapsed time: {t1 - t0:.2f}s")

def parse_args():
    parser = argparse.ArgumentParser(description="Run policy rollouts with vLLM")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--old_policy", action="store_true", help="Use old policy")
    parser.add_argument("--start_id", type=int, required=True, help="Starting index in dataset")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to process")
    parser.add_argument("--max_tokens", type=int, required=True, help="Maximum tokens to generate")
    parser.add_argument("--rollout_size", type=int, required=True, help="Number of completions per prompt")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for processing")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save output")
    return parser.parse_args()

if __name__ == "__main__":
    os.environ['TQDM_DISABLE'] = '1'
    args = parse_args()
    main(
        model_name=args.model_name,
        policy_path=args.policy_path,
        gpu_id=args.gpu_id,
        dataset_path=args.dataset_path,
        old_policy=args.old_policy,
        start_id=args.start_id,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        rollout_size=args.rollout_size,
        batch_size=args.batch_size,
        save_path=args.save_path,
    )
