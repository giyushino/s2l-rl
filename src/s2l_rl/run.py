import os
import subprocess
from pathlib import Path

def split_dataset(total_examples: int, num_gpus: int):
    """Split dataset into chunks for each GPU"""
    base_size = total_examples // num_gpus
    remainder = total_examples % num_gpus

    splits = []
    start_id = 0
    for i in range(num_gpus):
        # Add 1 extra example to first 'remainder' processes
        num_samples = base_size + (1 if i < remainder else 0)
        splits.append((start_id, num_samples))
        start_id += num_samples

    return splits

def main():
    # Configuration
    project_path = Path(__file__).parent.parent.parent.absolute()
    num_gpus = 8
    total_examples = 7473

    # Common arguments
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    policy_path = f"{project_path}/checkpoints/global_step_10/actor/lora_adapter/"
    dataset_path = f"{project_path}/datasets/gsm8k/train.parquet"
    old_policy = False
    max_tokens = 2048
    rollout_size = 16
    batch_size = 4
    save_dir = f"{project_path}/data/step_10"

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Split the dataset
    splits = split_dataset(total_examples, num_gpus)

    # Launch processes
    processes = []
    for gpu_id, (start_id, num_samples) in enumerate(splits):
        save_path = f"{save_dir}/output_{gpu_id}.jsonl"

        # Build command to run policy.py with argparse
        cmd = [
            "python", "-m", "s2l_rl.policy",
            "--model_name", model_name,
            "--policy_path", policy_path,
            "--gpu_id", str(gpu_id),
            "--dataset_path", dataset_path,
            "--start_id", str(start_id),
            "--num_samples", str(num_samples),
            "--max_tokens", str(max_tokens),
            "--rollout_size", str(rollout_size),
            "--batch_size", str(batch_size),
            "--save_path", save_path,
        ]

        # Add --old_policy flag if True
        if old_policy:
            cmd.append("--old_policy")

        print(f"Launching GPU {gpu_id}: start_id={start_id}, num_samples={num_samples}")
        process = subprocess.Popen(cmd)
        processes.append(process)

    # Wait for all processes to complete
    print(f"\nWaiting for {num_gpus} processes to complete...")
    for i, process in enumerate(processes):
        process.wait()
        print(f"GPU {i} completed with exit code {process.returncode}")

    print("\nAll processes completed!")

if __name__ == "__main__":
    main()
