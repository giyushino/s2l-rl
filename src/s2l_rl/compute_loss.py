#conda_env: s2l
import os
import time
import orjson
import argparse
import numpy as np
from typing import List
from concurrent.futures import ProcessPoolExecutor

def load_single_file(file_path: str) -> list[dict]:
    """Load a single JSONL file - runs in separate process."""
    with open(file_path, "rb") as file:
        return [orjson.loads(line) for line in file]

def load_data(folder_path:str) -> List[dict]:
    print("loading data...")
    t0 = time.perf_counter()
    file_paths = [os.path.join(folder_path, f"output_{x}.jsonl") for x in range(8)]

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_single_file, file_paths)

    data =  [item for file_data in results for item in file_data]
    t1 = time.perf_counter()
    print(f"time to load data: {t1 - t0}")
    print(len(data))
    print(len(data)/16)

    return data

def compute_grpo_loss(batch: List[dict], clip_eps: float = 0.2, beta: float = 0.01) -> float:
    old_logprobs = []
    logprobs = []
    rewards = []
    for traj in batch:
        old_logprobs.append(traj["old_logprobs"])
        logprobs.append(traj["log_probs"])
        rewards.append(traj["model_answer"] == traj["ground_truth"])

    # unless we want to mask, we can't directly turn this into a tensor
    for i in range(16):
        old_logprobs[i] = sum(old_logprobs[i])
        logprobs[i] = sum(logprobs[i])

    # cast everything to numpy array
    rewards = np.array(rewards)
    logprobs = np.array(logprobs)
    old_logprobs = np.array(old_logprobs)

    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)    
    ratio = np.exp(logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -np.minimum(surr1, surr2).mean()

    # we can do an approximation of KL loss with the previous step
    return policy_loss.item()

def main(step_id: int, folder_path: str, batch_size: int, save_folder: str):
    # Load all data from the folder
    data = load_data(folder_path)

    # Compute losses for each batch
    losses = []
    num_batches = len(data) // batch_size

    print(f"\nComputing GRPO losses for {num_batches} batches...")
    for i in range(num_batches):
        batch = data[i * batch_size : (i + 1) * batch_size]
        loss = compute_grpo_loss(batch)
        losses.append(loss)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"Batch {i + 1}/{num_batches}, Loss: {loss:.4f}")

    # Convert to numpy array for statistics
    losses = np.array(losses)

    # Create save directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Save losses to file
    save_path = os.path.join(save_folder, f"losses_{step_id}.npy")
    np.save(save_path, losses)
    print(f"\nSaved losses to: {save_path}")

    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"GRPO Loss Statistics:")
    print(f"{'='*50}")
    print(f"Mean Loss:   {losses.mean():.4f}")
    print(f"Std Loss:    {losses.std():.4f}")
    print(f"Min Loss:    {losses.min():.4f}")
    print(f"Max Loss:    {losses.max():.4f}")
    print(f"Median Loss: {np.median(losses):.4f}")
    print(f"{'='*50}")

    return losses


def parse_args():
    parser = argparse.ArgumentParser(description="Compute GRPO losses for a dataset")
    parser.add_argument("--step_id", type=int, required=True, help="Step ID for naming the output file")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to folder containing JSONL files")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for loss computation")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to folder for saving losses")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        step_id=args.step_id,
        folder_path=args.folder_path,
        batch_size=args.batch_size,
        save_folder=args.save_folder,
    )
