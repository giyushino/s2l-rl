"""
S2L: Sample selection based on loss trajectory clustering
Uses k-means clustering on loss trajectories to select diverse samples
"""
import numpy as np
import glob
import time
import os
from typing import List, Optional


class S2L:
    def __init__(
        self,
        loss_folder: str,
        n_components: int = 50,
        verbose: bool = True
    ):
        """
        Initialize S2L with loss trajectories from multiple checkpoints

        Args:
            loss_folder: Path to folder containing losses_*.npy files
            n_components: Number of k-means clusters for diversity sampling
            verbose: Whether to print progress information
        """
        self.n_components = n_components
        self.verbose = verbose
        self.loss_folder = loss_folder

        # Load all loss files
        self.losses = self._load_losses()
        self.n_samples = self.losses.shape[0]

        if self.verbose:
            print(f"Loaded {self.losses.shape[1]} checkpoints with {self.n_samples} samples")
            print(f"Loss trajectory shape: {self.losses.shape}")

    def _load_losses(self) -> np.ndarray:
        """Load all loss files and stack them into a matrix"""
        loss_files = sorted(glob.glob(os.path.join(self.loss_folder, "losses_*.npy")))

        if len(loss_files) == 0:
            raise ValueError(f"No loss files found in {self.loss_folder}")

        losses = []
        for loss_file in loss_files:
            if self.verbose:
                print(f"Loading {loss_file}...")
            loss = np.load(loss_file)
            losses.append(loss)

        # Stack losses: shape (n_samples, n_checkpoints)
        losses = np.stack(losses, axis=1)

        # Replace NaN with 0
        losses[np.isnan(losses)] = 0

        return losses

    def select_diverse_samples(
        self,
        n: int,
        indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select n diverse samples using k-means clustering on loss trajectories

        Args:
            n: Number of samples to select
            indices: Optional array of indices to select from (if None, use all samples)

        Returns:
            Array of selected indices
        """
        if indices is None:
            indices = np.arange(self.n_samples)

        if len(indices) <= n:
            if self.verbose:
                print(f"Requested {n} samples but only {len(indices)} available, returning all")
            return indices

        # Get loss trajectories for the specified indices
        features = self.losses[indices]

        # Perform k-means clustering
        selected = self._faiss_kmeans_selection(features, n)

        return indices[selected]

    def _faiss_kmeans_selection(self, features: np.ndarray, n: int) -> np.ndarray:
        """
        K-means selection for diverse sampling

        Args:
            features: Feature matrix (n_samples, n_features)
            n: Number of samples to select

        Returns:
            Array of selected indices
        """
        import faiss

        start_time = time.time()

        # Use min of requested components and number of samples
        n_clusters = min(self.n_components, len(features))

        if self.verbose:
            print(f"Running k-means with {n_clusters} clusters...")

        kmeans = faiss.Kmeans(
            features.shape[1],
            n_clusters,
            niter=20,
            verbose=self.verbose
        )
        kmeans.train(features.astype(np.float32))

        # Get cluster assignments
        D, I = kmeans.index.search(features.astype(np.float32), 1)
        I = I.flatten()

        if self.verbose:
            print(f"K-means clustering took {time.time() - start_time:.2f} seconds")

        # Get cluster sizes
        clusters, counts = np.unique(I, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]  # Sort by size (largest first)

        # Filter out very small clusters (size <= 2)
        large_clusters = sorted_idx[counts[sorted_idx] > 2]
        small_clusters = sorted_idx[counts[sorted_idx] <= 2]

        if self.verbose:
            print(f"Found {len(large_clusters)} clusters with size > 2")
            print(f"Found {len(small_clusters)} clusters with size <= 2")

        sampled_indices = []

        # Sample from large clusters first (proportionally)
        for i, cluster_idx in enumerate(large_clusters):
            n_remaining_clusters = len(large_clusters) - i
            n_per_cluster = n // n_remaining_clusters

            cluster_indices = np.where(I == clusters[cluster_idx])[0]

            if len(cluster_indices) > n_per_cluster:
                sampled_indices.append(
                    np.random.choice(cluster_indices, n_per_cluster, replace=False)
                )
                n -= n_per_cluster
            else:
                sampled_indices.append(cluster_indices)
                n -= len(cluster_indices)

        # If we still need more samples, sample from small clusters
        if n > 0 and len(small_clusters) > 0:
            if self.verbose:
                print(f"Sampling {n} additional samples from small clusters")

            small_cluster_indices = np.where(np.isin(I, clusters[small_clusters]))[0]

            if len(small_cluster_indices) >= n:
                sampled_indices.append(
                    np.random.choice(small_cluster_indices, n, replace=False)
                )
            else:
                sampled_indices.append(small_cluster_indices)

        return np.concatenate(sampled_indices)


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Select diverse samples using S2L")
    parser.add_argument("--loss_folder", type=str, required=True,
                       help="Path to folder containing loss files")
    parser.add_argument("--n_samples", type=int, required=True,
                       help="Number of samples to select")
    parser.add_argument("--n_components", type=int, default=50,
                       help="Number of k-means clusters")
    parser.add_argument("--save_path", type=str, default=None,
                       help="Path to save selected indices")
    args = parser.parse_args()

    # Initialize S2L
    s2l = S2L(
        loss_folder=args.loss_folder,
        n_components=args.n_components,
        verbose=True
    )

    # Select diverse samples
    selected_indices = s2l.select_diverse_samples(args.n_samples)

    print(f"\nSelected {len(selected_indices)} diverse samples")
    print(f"Indices: {selected_indices[:10]}..." if len(selected_indices) > 10 else f"Indices: {selected_indices}")

    # Save if requested
    if args.save_path:
        np.save(args.save_path, selected_indices)
        print(f"Saved selected indices to {args.save_path}")


if __name__ == "__main__":
    main()
