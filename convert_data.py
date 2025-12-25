#conda_env: s2l
import os
import json
from datasets import Dataset, load_dataset

import argparse

CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_PATH)

def chat_template(question):
    prompt = "<|im_start|>system\nPlease reason step by step, and present the answer in LaTex format: \\boxed{Your answer}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def custom_load(mode, data_path = "/home/allanz/omega/data/easy_poly_v8.jsonl", turn_off_thinking=False):
    data = []
    with open(data_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            temp_data = json.loads(line)
            print(temp_data)
            prompt = "<|im_start|>system\nPlease reason step by step, and present the answer in LaTex format: \\boxed{Your answer}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{temp_data['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            temp_data['prompt'] = prompt
            data.append(temp_data)

    if mode == "train":
        return Dataset.from_list(data).select(range(900))
    if mode == "eval":
        return Dataset.from_list(data).select(range(900, 1000))


def reformat_dataset(dataset, turn_off_thinking=False):
    """
    Reformat the oob dataset to work with grpo trainer class.

    Args:
        dataset: Primary dataset to reformat
        turn_off_thinking: If True, add /no_think tags to disable thinking mode
    Returns:
        Dataset with reformatted prompts
    """
    reformatted = []
    count = 0

    # First pass: collect all valid samples
    all_samples = []
    
    for element in dataset: 
        try: 
            # extract question and remove their formatting instructions
            question = element["question"]
            num_str = element["answer"].split("####")[-1]
            num = int(num_str.replace(",", ""))
            all_samples.append({
                "question": question,
                "ground_truth": num
            })
        except: 
            count += 1

    print(f"{count} of {count + len(all_samples)} elements were skipped due to formatting issues")
    
    # Format evaluation samples with optional few-shot examples
    if turn_off_thinking:
        for sample in all_samples:
            prompt = "<|im_start|>system\nPlease reason step by step, and present the answer in LaTex format: \\boxed{Your answer}<|im_end|>\n"
            prompt += f"<|im_start|>user\n/no_think\n{sample['question']}\n/no_think<|im_end|>\n<|im_start|>assistant\n"

            reformatted.append({
                "prompt": prompt,
                "ground_truth": sample["ground_truth"]
            })
    else:
        for sample in all_samples:
            prompt = "<|im_start|>system\nPlease reason step by step, and present the answer in LaTex format: \\boxed{Your answer}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n"

            reformatted.append({
                "prompt": prompt,
                "ground_truth": sample["ground_truth"]
            })

    return Dataset.from_list(reformatted).shuffle(seed=42)

def load_gsm8k(mode, turn_off_thinking=False):
    if mode == "train":
        return reformat_dataset(load_dataset("openai/gsm8k", "main")["train"], turn_off_thinking)
    if mode == "eval":
        return reformat_dataset(load_dataset("openai/gsm8k", "main")["test"], turn_off_thinking)
    else:
        print("non-valid mode passed, can only be either train, eval")


if __name__ == '__main__':
    # example code for turing gsm8k into parquet file for verl training

    if not os.path.isdir(os.path.join(PROJECT_ROOT, "datasets")):
        os.makedirs(os.path.join(PROJECT_ROOT, "datasets"))

    dataset_name = "gsm8k"
    dataset_save_path = os.path.join(PROJECT_ROOT, "datasets", dataset_name)

    #train_dataset = load_dataset("hiyouga/math12k")["train"]
    #test_dataset = load_dataset("hiyouga/math12k")["test"]
    train_dataset = load_gsm8k("train")
    test_dataset = load_gsm8k("eval")

    print(train_dataset)

    # Construct a `def make_map_fn(split)` for the corresponding datasets.
    def make_map_fn(split):
        def process_fn(example, idx):
            #question = chat_template(example["problem"])
            #answer = example["answer"]
            question = example["prompt"]
            # make this a string or else when we grade it won't work

            answer = str(example["ground_truth"])
            data = {
                "data_source": f"{dataset_name}",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(dataset_save_path, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(dataset_save_path, 'test.parquet'))
    print(f"datasets saved to {dataset_save_path}")
