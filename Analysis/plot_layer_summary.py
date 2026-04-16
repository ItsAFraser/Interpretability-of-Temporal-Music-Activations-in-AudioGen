import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Temporal-Music-Activations"))

from SparseAutoencoder import SparseAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate three summary figures for one SAE layer: "
            "single-track top-k time series, single-track heatmap, "
            "and an averaged heatmap across many tracks."
        )
    )
    parser.add_argument("--features_root", type=str, required=True, help="Root directory containing layer_*/ feature files.")
    parser.add_argument("--sae_root", type=str, required=True, help="Root directory containing trained SAE layer subdirectories.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where figures will be saved.")
    parser.add_argument("--layer_name", type=str, default="layer_20", help="Layer subdirectory to analyze, e.g. layer_20.")
    parser.add_argument("--track_index", type=int, default=0, help="Track index for single-track plots.")
    parser.add_argument("--avg_tracks", type=int, default=100, help="How many tracks to average for the summary heatmap.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top features to plot in the single-track time-series figure.")
    parser.add_argument("--heatmap_k", type=int, default=256, help="Number of features to show in heatmaps.")
    parser.add_argument("--time_bins", type=int, default=300, help="Number of bins when resampling time for plotting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when subsampling tracks.")
    return parser.parse_args()


def load_sae(layer_dir: Path) -> SparseAutoencoder:
    checkpoint_candidates = [layer_dir / "sae_best.pt", layer_dir / "sae_final.pt"]
    ckpt_path = next((path for path in checkpoint_candidates if path.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"No SAE checkpoint found in {layer_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    input_dim = state["encoder.weight"].shape[1]
    latent_dim = state["encoder.weight"].shape[0]

    model = SparseAutoencoder(input_dim, latent_dim).cpu()
    model.load_state_dict(state)
    model.eval()
    return model


def collect_feature_paths(layer_features_dir: Path) -> list[Path]:
    return sorted(layer_features_dir.rglob("*.npy"))


def encode_track(model: SparseAutoencoder, npy_path: Path) -> np.ndarray:
    features = np.load(npy_path)
    expected_dim = model.encoder.weight.shape[1]
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected feature shape for {npy_path}: {features.shape}, expected [T, {expected_dim}]"
        )

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        acts = model.activation(model.encoder(x))
    return acts.cpu().numpy()


def normalize_series(series: np.ndarray) -> np.ndarray:
    return (series - series.min()) / (series.max() - series.min() + 1e-12)


def resample_feature_matrix(matrix: np.ndarray, n_bins: int) -> np.ndarray:
    old_x = np.linspace(0, 1, matrix.shape[0])
    new_x = np.linspace(0, 1, n_bins)
    return np.stack(
        [np.interp(new_x, old_x, matrix[:, feature_idx]) for feature_idx in range(matrix.shape[1])],
        axis=1,
    )


def save_single_track_topk_timeseries(
    acts: np.ndarray,
    output_path: Path,
    top_k: int,
) -> None:
    feature_peaks = acts.max(axis=0)
    top_indices = np.argsort(feature_peaks)[::-1][:top_k]
    x_axis = np.linspace(0, 1, acts.shape[0])

    fig, axes = plt.subplots(top_k, 1, figsize=(10, 2.2 * top_k), sharex=True)
    if top_k == 1:
        axes = [axes]

    for ax, feature_idx in zip(axes, top_indices):
        ts = normalize_series(acts[:, feature_idx])
        ax.plot(x_axis, ts, color="#C65F5F", linewidth=1.5)
        ax.fill_between(x_axis, ts, alpha=0.15, color="#C65F5F")
        ax.set_ylabel(f"f{feature_idx}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.15, linewidth=0.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("Relative time")
    fig.suptitle(f"Top {top_k} SAE Features Over Time", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_single_track_heatmap(
    acts: np.ndarray,
    output_path: Path,
    heatmap_k: int,
    time_bins: int,
) -> None:
    feature_peaks = acts.max(axis=0)
    top_indices = np.argsort(feature_peaks)[::-1][:min(heatmap_k, acts.shape[1])]
    resampled = resample_feature_matrix(acts[:, top_indices], time_bins).T
    resampled = resampled / (resampled.max(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(12, 6))
    plt.imshow(resampled, aspect="auto", interpolation="nearest", cmap="viridis", origin="upper")
    plt.colorbar(label="Normalized activation")
    plt.xlabel("Relative time")
    plt.ylabel(f"Top {resampled.shape[0]} SAE features")
    plt.title("Single-Track Activation Heatmap")
    plt.xticks(
        np.linspace(0, resampled.shape[1] - 1, 6),
        [f"{int(v)}%" for v in np.linspace(0, 100, 6)],
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_average_heatmap(
    model: SparseAutoencoder,
    npy_files: list[Path],
    output_path: Path,
    avg_tracks: int,
    heatmap_k: int,
    time_bins: int,
    seed: int,
) -> None:
    if not npy_files:
        raise ValueError("No feature files available for averaged heatmap.")

    rng = np.random.default_rng(seed)
    if len(npy_files) > avg_tracks:
        sampled_indices = np.sort(rng.choice(len(npy_files), size=avg_tracks, replace=False))
        sampled_paths = [npy_files[int(index)] for index in sampled_indices]
    else:
        sampled_paths = npy_files

    peak_sums = None
    resampled_tracks = []

    for path in sampled_paths:
        acts = encode_track(model, path)
        peaks = acts.max(axis=0)
        if peak_sums is None:
            peak_sums = np.zeros_like(peaks)
        peak_sums += peaks
        resampled_tracks.append(resample_feature_matrix(acts, time_bins))

    assert peak_sums is not None
    top_indices = np.argsort(peak_sums)[::-1][:min(heatmap_k, peak_sums.shape[0])]
    stacked = np.stack([track[:, top_indices] for track in resampled_tracks], axis=0)
    mean_heatmap = stacked.mean(axis=0).T
    mean_heatmap = mean_heatmap / (mean_heatmap.max(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(12, 6))
    plt.imshow(mean_heatmap, aspect="auto", interpolation="nearest", cmap="viridis", origin="upper")
    plt.colorbar(label="Mean normalized activation")
    plt.xlabel("Relative time")
    plt.ylabel(f"Top {mean_heatmap.shape[0]} SAE features")
    plt.title(f"Averaged Activation Heatmap Across {len(sampled_paths)} Tracks")
    plt.xticks(
        np.linspace(0, mean_heatmap.shape[1] - 1, 6),
        [f"{int(v)}%" for v in np.linspace(0, 100, 6)],
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    features_root = Path(args.features_root).expanduser().resolve()
    sae_root = Path(args.sae_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_features_dir = features_root / args.layer_name
    layer_model_dir = sae_root / args.layer_name

    if not layer_features_dir.exists():
        raise FileNotFoundError(f"Feature layer directory not found: {layer_features_dir}")
    if not layer_model_dir.exists():
        raise FileNotFoundError(f"SAE layer directory not found: {layer_model_dir}")

    model = load_sae(layer_model_dir)
    npy_files = collect_feature_paths(layer_features_dir)
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {layer_features_dir}")
    if args.track_index < 0 or args.track_index >= len(npy_files):
        raise IndexError(f"track_index={args.track_index} is out of range for {len(npy_files)} tracks")

    selected_track_path = npy_files[args.track_index]
    selected_track_acts = encode_track(model, selected_track_path)

    out_timeseries = output_dir / f"{args.layer_name}_single_track_top{args.top_k}_timeseries_track{args.track_index}.png"
    out_heatmap = output_dir / f"{args.layer_name}_single_track_heatmap_top{args.heatmap_k}_track{args.track_index}.png"
    out_avg_heatmap = output_dir / f"{args.layer_name}_average_heatmap_top{args.heatmap_k}_{min(args.avg_tracks, len(npy_files))}tracks.png"

    save_single_track_topk_timeseries(selected_track_acts, out_timeseries, args.top_k)
    save_single_track_heatmap(selected_track_acts, out_heatmap, args.heatmap_k, args.time_bins)
    save_average_heatmap(
        model=model,
        npy_files=npy_files,
        output_path=out_avg_heatmap,
        avg_tracks=args.avg_tracks,
        heatmap_k=args.heatmap_k,
        time_bins=args.time_bins,
        seed=args.seed,
    )

    print("Saved figures:")
    print(out_timeseries)
    print(out_heatmap)
    print(out_avg_heatmap)
    print(f"Single-track source: {selected_track_path}")


if __name__ == "__main__":
    main()
