"""Report ImageNet clustering diagnostics for the RQ4 min_samples analysis.

The script reproduces the clustering pipeline of CMPS++: it reads source
probability vectors, applies StandardScaler, and invokes ``my_dbscan.dbscan``
with cosine distance and epsilon=0.15. It then reports cluster counts, noise,
cluster-size summaries, and intra-/inter-cluster cosine-distance summaries for
min_samples in {1, ..., 5}.

Run from the project root:
    python Source_code/cluster_diagnostics.py

The output is saved to:
    Experiment_results/RQ4/imagenet_cluster_diagnostics.csv
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

import my_dbscan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "Input_data"
OUTPUT_FILE = PROJECT_ROOT / "Experiment_results" / "RQ4" / "imagenet_cluster_diagnostics.csv"

EPSILON = 0.15
MIN_SAMPLES_VALUES = range(1, 6)
MAX_INTER_CLUSTER_PAIRS = 10_000
RANDOM_SEED = 20251106
SUBJECTS = ("imagenet_googlenet", "imagenet_resnet")


def load_source_probabilities(subject: str) -> np.ndarray:
    """Load source probabilities exactly as in cmps_extended.data_process."""
    mat_file = INPUT_ROOT / subject / "flipLeftRight_classification_results.mat"
    data = loadmat(mat_file)
    source_confidences = []
    for item in data["confidenceSourceArray"][()]:
        source_confidences.append(np.asarray(item[0][0], dtype=float).ravel())
    # This is the same ImageNet dimensionality handling used by CMPS++.
    return np.asarray([np.pad(vector, (0, 1000 - len(vector)), "constant")
                       for vector in source_confidences], dtype=float)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.dot(left, right) / denominator)


def mean_intra_cluster_distance(points: np.ndarray, labels: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Mean point-to-own-centroid cosine distance, weighted by cluster size."""
    total_distance = 0.0
    total_points = 0
    for cluster_id in cluster_ids:
        members = points[labels == cluster_id]
        centroid = members.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        member_norms = np.linalg.norm(members, axis=1)
        valid = (member_norms > 0) & (centroid_norm > 0)
        distances = 1.0 - (members[valid] @ centroid) / (member_norms[valid] * centroid_norm)
        total_distance += float(distances.sum())
        total_points += len(distances)
    return total_distance / total_points if total_points else float("nan")


def mean_inter_cluster_distance(points: np.ndarray, labels: np.ndarray, cluster_ids: np.ndarray) -> tuple[float, int]:
    """Mean cosine distance between cluster centroids.

    All centroid pairs are used when there are at most 10,000; otherwise a
    deterministic uniform sample of 10,000 distinct pairs is used.
    """
    centroids = np.asarray([points[labels == cluster_id].mean(axis=0) for cluster_id in cluster_ids])
    cluster_count = len(centroids)
    all_pair_count = cluster_count * (cluster_count - 1) // 2
    if all_pair_count == 0:
        return float("nan"), 0

    if all_pair_count <= MAX_INTER_CLUSTER_PAIRS:
        pairs = list(itertools.combinations(range(cluster_count), 2))
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        selected = set()
        while len(selected) < MAX_INTER_CLUSTER_PAIRS:
            first, second = sorted(rng.choice(cluster_count, size=2, replace=False).tolist())
            selected.add((first, second))
        pairs = list(selected)

    distances = [cosine_distance(centroids[first], centroids[second]) for first, second in pairs]
    distances = [distance for distance in distances if not np.isnan(distance)]
    return (float(np.mean(distances)) if distances else float("nan")), len(distances)


def composite_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """The harmonic-mean DeepGini/MaxP score used by CMPS++."""
    deepgini = 1.0 - np.sum(np.square(probabilities), axis=1)
    maxp = 1.0 - np.max(probabilities, axis=1)
    return 2.0 * deepgini * maxp / (deepgini + maxp + 1e-10)


def diagnose(subject: str, probabilities: np.ndarray, min_samples: int) -> dict[str, object]:
    scaled = StandardScaler().fit_transform(probabilities)
    labels = my_dbscan.dbscan(scaled, eps=EPSILON, min_samples=min_samples)
    cluster_ids = np.asarray(sorted(cluster_id for cluster_id in set(labels) if cluster_id != -1))
    cluster_sizes = np.asarray([np.sum(labels == cluster_id) for cluster_id in cluster_ids], dtype=int)
    uncertainty = composite_uncertainty(probabilities)
    cluster_max_uncertainty = np.asarray([
        uncertainty[labels == cluster_id].max() for cluster_id in cluster_ids
    ])
    noise_count = int(np.sum(labels == -1))
    inter_distance, pair_count = mean_inter_cluster_distance(scaled, labels, cluster_ids)

    return {
        "Subject": subject,
        "Epsilon": EPSILON,
        "Min_Samples": min_samples,
        "Points": len(labels),
        "Clusters": len(cluster_ids),
        "Noise_Points": noise_count,
        "Noise_Proportion": noise_count / len(labels),
        "Cluster_Size_Min": int(cluster_sizes.min()) if len(cluster_sizes) else 0,
        "Cluster_Size_Median": float(np.median(cluster_sizes)) if len(cluster_sizes) else float("nan"),
        "Cluster_Size_IQR": float(np.percentile(cluster_sizes, 75) - np.percentile(cluster_sizes, 25)) if len(cluster_sizes) else float("nan"),
        "Cluster_Size_Max": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
        "Cluster_Max_Uncertainty_Mean": float(np.mean(cluster_max_uncertainty)) if len(cluster_max_uncertainty) else float("nan"),
        "Cluster_Max_Uncertainty_Median": float(np.median(cluster_max_uncertainty)) if len(cluster_max_uncertainty) else float("nan"),
        "Mean_Intra_Cluster_Distance": mean_intra_cluster_distance(scaled, labels, cluster_ids),
        "Mean_Inter_Cluster_Centroid_Distance": inter_distance,
        "Inter_Cluster_Pairs_Used": pair_count,
    }


def main() -> None:
    rows = []
    for subject in SUBJECTS:
        print(f"Loading source probabilities for {subject}...")
        probabilities = load_source_probabilities(subject)
        for min_samples in MIN_SAMPLES_VALUES:
            print(f"  min_samples={min_samples}")
            rows.append(diagnose(subject, probabilities, min_samples))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved diagnostics to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
