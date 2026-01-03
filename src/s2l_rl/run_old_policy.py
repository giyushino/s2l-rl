import os
import sys
import argparse
import subprocess
from pathlib import Path

def main(
    input_dir: str,
    policy_path: str,
    model_name: str,
    batch_size: int,
    num_gpus: int,
    gpu_memory_utilization: float = 0.8
):
    # Convert input_dir to Path object
    input_path = Path(input_dir)

    # Find all .jsonl files in the directory
    jsonl_files = sorted(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {input_dir}")
        return

    print(f"Found {len(jsonl_files)} .jsonl files to process")

    # Create complete/ subdirectory for outputs
    output_dir = input_path / "complete"
    os.makedirs(output_dir, exist_ok=True)

    # Launch processes for each file
    processes = []
    for idx, jsonl_file in enumerate(jsonl_files):
        # Assign GPU in round-robin fashion
        gpu_id = idx % num_gpus

        # Set up output path in complete/ subdirectory
        save_path = output_dir / jsonl_file.name

        # Build command to run old_policy.py with argparse
        cmd = [
            "python", "-m", "s2l_rl.old_policy",
            "--model_name", model_name,
            "--policy_path", policy_path,
            "--gpu_id", str(gpu_id),
            "--data_path", str(jsonl_file),
            "--save_path", str(save_path),
            "--batch_size", str(batch_size),
            "--gpu_memory_utilization", str(gpu_memory_utilization),
        ]

        # Enable verbose logging only for GPU 0
        if gpu_id == 0:
            cmd.append("--verbose")

        print(f"Launching process for {jsonl_file.name} on GPU {gpu_id}")
        process = subprocess.Popen(cmd)
        processes.append((process, jsonl_file.name, gpu_id))

    # Wait for all processes to complete
    print(f"\nWaiting for {len(processes)} processes to complete...")
    for process, filename, gpu_id in processes:
        process.wait()
        print(f"File {filename} (GPU {gpu_id}) completed with exit code {process.returncode}")

    print("\nAll processes completed!")

def parse_args():
    parser = argparse.ArgumentParser(description="Run old_policy.py on multiple JSONL files in parallel")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSONL files to process")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to old policy LoRA adapter")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for processing")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization (default: 0.8)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        input_dir=args.input_dir,
        policy_path=args.policy_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
