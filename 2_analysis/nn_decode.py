import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import pairwise_distances


def load_session_data(file_path: Path) -> pd.DataFrame:
    """Load a single session .h5 file."""
    return pd.read_hdf(file_path, key="df")


def prepare_repeats(df: pd.DataFrame):
    """Organize first and second repeats for each stimulus.

    Returns
    -------
    rep1_matrix: ndarray of shape (n_images, n_features)
    rep2_matrix: ndarray of shape (n_images, n_features)
    labels: list of stimulus ids corresponding to rows
    """
    rep1_list = []
    rep2_list = []
    labels = []
    for stim_id, grp in df.groupby("73k_id"):
        # require at least two trials
        if len(grp) < 2:
            continue
        # sort by trial order so that the first and second appearance are used
        grp = grp.sort_values("trial")
        first, second = grp.iloc[0], grp.iloc[1]
        rep1_list.append(first["eeg"].ravel())
        rep2_list.append(second["eeg"].ravel())
        labels.append(stim_id)
    rep1_matrix = np.vstack(rep1_list)
    rep2_matrix = np.vstack(rep2_list)
    return rep1_matrix, rep2_matrix, labels


def one_nn_accuracy(rep1_matrix: np.ndarray, rep2_matrix: np.ndarray, labels):
    """Compute 1-NN classification accuracy between two repetitions."""
    dists = pairwise_distances(rep1_matrix, rep2_matrix, metric="euclidean")
    nearest = dists.argmin(axis=1)
    preds = [labels[i] for i in nearest]
    acc = np.mean([preds[i] == labels[i] for i in range(len(labels))])
    return acc


def main():
    parser = argparse.ArgumentParser(description="Nearest neighbor benchmark")
    parser.add_argument("--dataset_dir", type=Path, required=True,
                        help="Path to folder with h5 files")
    parser.add_argument("--freq_band", default="05_125",
                        help="Frequency band folder inside dataset_dir")
    args = parser.parse_args()

    pattern = args.dataset_dir / "final_hdf5" / args.freq_band / "*.h5"
    files = sorted(args.dataset_dir.joinpath("final_hdf5", args.freq_band).glob("*.h5"))

    if not files:
        raise FileNotFoundError(f"No .h5 files found in {pattern}")

    for f in files:
        df = load_session_data(f)
        rep1, rep2, labels = prepare_repeats(df)
        acc = one_nn_accuracy(rep1, rep2, labels)
        print(f"{f.name}: {acc:.3f}")


if __name__ == "__main__":
    main()
