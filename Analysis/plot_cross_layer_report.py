import argparse
import json
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
            "Generate three cross-layer report figures: averaged heatmap panel, "
            "temporal-statistics summary, and representative feature gallery."
        )
    )
    parser.add_argument("--features_root", type=str, required=True, help="Root directory containing extracted feature layers.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where figures and metadata will be saved.")
    parser.add_argument(
        "--layer_spec",
        action="append",
        required=True,
        help=(
            "Layer-to-SAE mapping in the form layer_name=/path/to/layer_model_dir. "
            "Pass multiple times, e.g. --layer_spec layer_20=/.../layer_20"
        ),
    )
    parser.add_argument("--avg_tracks", type=int, default=100, help="How many common tracks to sample across layers.")
    parser.add_argument("--heatmap_k", type=int, default=256, help="Number of top-ranked features shown in averaged heatmaps.")
    parser.add_argument("--time_bins", type=int, default=300, help="Temporal bins after resampling to relative time.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic track subsampling.")
    return parser.parse_args()


def parse_layer_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --layer_spec '{spec}'. Expected layer_name=/path/to/model_dir")
        layer_name, path_str = spec.split("=", 1)
        layer_name = layer_name.strip()
        model_dir = Path(path_str.strip()).expanduser().resolve()
        if not layer_name:
            raise ValueError(f"Invalid --layer_spec '{spec}': empty layer name")
        if layer_name in seen:
            raise ValueError(f"Duplicate layer_name in --layer_spec: {layer_name}")
        seen.add(layer_name)
        parsed.append((layer_name, model_dir))
    return parsed


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


def collect_relative_feature_map(layer_features_dir: Path) -> dict[str, Path]:
    return {
        path.relative_to(layer_features_dir).as_posix(): path
        for path in sorted(layer_features_dir.rglob("*.npy"))
    }


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


def resample_feature_matrix(matrix: np.ndarray, n_bins: int) -> np.ndarray:
    old_x = np.linspace(0, 1, matrix.shape[0])
    new_x = np.linspace(0, 1, n_bins)
    return np.stack(
        [np.interp(new_x, old_x, matrix[:, feature_idx]) for feature_idx in range(matrix.shape[1])],
        axis=1,
    ).astype(np.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_max = matrix.max(axis=1, keepdims=True)
    return matrix / (row_max + 1e-12)


def autocorr_peak(series: np.ndarray) -> float:
    if series.std() < 1e-8:
        return 0.0
    s = series - series.mean()
    full = np.correlate(s, s, mode="full")
    mid = len(full) // 2
    max_lag = min(len(series) // 2, 100)
    ac = full[mid:mid + max_lag + 1] / (full[mid] + 1e-12)
    return float(ac[1:].max()) if len(ac) > 1 else 0.0


def feature_statistics(resampled_tracks: list[np.ndarray]) -> dict[str, np.ndarray]:
    stacked = np.stack(resampled_tracks, axis=0)  # [tracks, time, features]
    active_frac = (stacked > 0).mean(axis=(0, 1))
    temporal_var = stacked.var(axis=1).mean(axis=0)

    n_features = stacked.shape[2]
    autocorr = np.zeros(n_features, dtype=np.float32)
    trend = np.zeros(n_features, dtype=np.float32)
    x = np.linspace(0, 1, stacked.shape[1], dtype=np.float32)
    for feature_idx in range(n_features):
        ac_values = []
        trend_values = []
        for track_idx in range(stacked.shape[0]):
            ts = stacked[track_idx, :, feature_idx]
            ac_values.append(autocorr_peak(ts))
            slope = np.polyfit(x, ts, 1)[0]
            trend_values.append(abs(float(slope)))
        autocorr[feature_idx] = float(np.mean(ac_values))
        trend[feature_idx] = float(np.mean(trend_values))

    return {
        "active_frac": active_frac,
        "temporal_var": temporal_var,
        "autocorr_peak": autocorr,
        "trend_slope": trend,
    }


def select_distinct_feature(primary: np.ndarray, secondary: np.ndarray | None = None, exclude: set[int] | None = None) -> int:
    exclude = exclude or set()
    order = np.argsort(primary)[::-1]
    for idx in order:
        idx_int = int(idx)
        if idx_int in exclude:
            continue
        if secondary is not None and not np.isfinite(secondary[idx_int]):
            continue
        return idx_int
    raise ValueError("Unable to select a representative feature.")


def save_average_heatmap_panel(
    layer_order: list[str],
    layer_heatmaps: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    n_layers = len(layer_order)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 2.8 * n_layers), sharex=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, layer_order):
        heatmap = layer_heatmaps[layer_name]
        image = ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="viridis", origin="upper")
        ax.set_title(layer_name, fontsize=12, pad=6)
        ax.set_ylabel(f"Top {heatmap.shape[0]} feats", fontsize=9)
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0, heatmap.shape[1] - 1, 6))
        ax.set_xticklabels([f"{int(v)}%" for v in np.linspace(0, 100, 6)], fontsize=8)

    axes[-1].set_xlabel("Relative time", fontsize=10)
    fig.suptitle("Cross-Layer Averaged Activation Heatmaps", fontsize=14, y=0.995)
    fig.colorbar(image, ax=axes, fraction=0.015, pad=0.01, label="Mean normalized activation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_temporal_stats_plot(
    layer_order: list[str],
    stats_summary: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    metrics = [
        ("active_frac", "Mean active fraction"),
        ("temporal_var", "Mean temporal variance"),
        ("autocorr_peak", "Mean autocorr. peak"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
    for ax, (metric_key, title) in zip(axes, metrics):
        values = [stats_summary[layer_name][metric_key] for layer_name in layer_order]
        ax.bar(layer_order, values, color="#4C72B0", edgecolor="white")
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=25)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle("Cross-Layer Temporal Summary Statistics", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_feature_gallery(
    layer_order: list[str],
    layer_resampled_tracks: dict[str, list[np.ndarray]],
    layer_stats: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> dict[str, dict[str, int]]:
    fig, axes = plt.subplots(len(layer_order), 2, figsize=(12, 3.2 * len(layer_order)), sharex=True)
    if len(layer_order) == 1:
        axes = np.array([axes])

    feature_choices: dict[str, dict[str, int]] = {}
    x_axis = np.linspace(0, 1, layer_resampled_tracks[layer_order[0]][0].shape[0])

    for row, layer_name in enumerate(layer_order):
        stats = layer_stats[layer_name]
        periodic_idx = select_distinct_feature(stats["autocorr_peak"])
        narrative_idx = select_distinct_feature(stats["trend_slope"], exclude={periodic_idx})
        feature_choices[layer_name] = {
            "periodic_feature": periodic_idx,
            "narrative_feature": narrative_idx,
        }

        stacked = np.stack(layer_resampled_tracks[layer_name], axis=0)  # [tracks, time, features]
        for col, (label, feature_idx, color) in enumerate([
            ("Most periodic", periodic_idx, "#DD8452"),
            ("Strongest trend", narrative_idx, "#55A868"),
        ]):
            ts = stacked[:, :, feature_idx].mean(axis=0)
            ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-12)
            ax = axes[row, col]
            ax.plot(x_axis, ts, color=color, linewidth=1.6)
            ax.fill_between(x_axis, ts, alpha=0.15, color=color)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.15, linewidth=0.5)
            ax.set_title(f"{label} | f{feature_idx}", fontsize=10)
            if col == 0:
                ax.set_ylabel(layer_name, fontsize=10, fontweight="bold")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    axes[-1, 0].set_xlabel("Relative time")
    axes[-1, 1].set_xlabel("Relative time")
    fig.suptitle("Representative Feature Gallery Across Layers", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return feature_choices


def main() -> None:
    args = parse_args()
    features_root = Path(args.features_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_specs = parse_layer_specs(args.layer_spec)
    layer_order = [layer_name for layer_name, _ in layer_specs]

    layer_rel_maps: dict[str, dict[str, Path]] = {}
    for layer_name, _ in layer_specs:
        layer_dir = features_root / layer_name
        if not layer_dir.exists():
            raise FileNotFoundError(f"Feature layer directory not found: {layer_dir}")
        layer_rel_maps[layer_name] = collect_relative_feature_map(layer_dir)
        if not layer_rel_maps[layer_name]:
            raise FileNotFoundError(f"No .npy files found in {layer_dir}")

    common_rel_paths = set(next(iter(layer_rel_maps.values())).keys())
    for rel_map in layer_rel_maps.values():
        common_rel_paths &= set(rel_map.keys())
    common_rel_paths = sorted(common_rel_paths)
    if not common_rel_paths:
        raise ValueError("No common feature files found across the requested layers.")

    rng = np.random.default_rng(args.seed)
    if len(common_rel_paths) > args.avg_tracks:
        sampled_indices = np.sort(rng.choice(len(common_rel_paths), size=args.avg_tracks, replace=False))
        sampled_rel_paths = [common_rel_paths[int(index)] for index in sampled_indices]
    else:
        sampled_rel_paths = common_rel_paths

    layer_heatmaps: dict[str, np.ndarray] = {}
    layer_resampled_tracks: dict[str, list[np.ndarray]] = {}
    layer_stats: dict[str, dict[str, np.ndarray]] = {}
    stats_summary: dict[str, dict[str, float]] = {}

    for layer_name, model_dir in layer_specs:
        model = load_sae(model_dir)
        resampled_tracks: list[np.ndarray] = []
        peak_sums = None

        for rel_path in sampled_rel_paths:
            npy_path = layer_rel_maps[layer_name][rel_path]
            acts = encode_track(model, npy_path)
            peaks = acts.max(axis=0)
            if peak_sums is None:
                peak_sums = np.zeros_like(peaks)
            peak_sums += peaks
            resampled_tracks.append(resample_feature_matrix(acts, args.time_bins))

        assert peak_sums is not None
        top_indices = np.argsort(peak_sums)[::-1][:min(args.heatmap_k, peak_sums.shape[0])]
        stacked = np.stack([track[:, top_indices] for track in resampled_tracks], axis=0)
        mean_heatmap = stacked.mean(axis=0).T
        layer_heatmaps[layer_name] = normalize_rows(mean_heatmap)
        layer_resampled_tracks[layer_name] = resampled_tracks

        stats = feature_statistics(resampled_tracks)
        layer_stats[layer_name] = stats
        stats_summary[layer_name] = {
            "active_frac": float(np.mean(stats["active_frac"])),
            "temporal_var": float(np.mean(stats["temporal_var"])),
            "autocorr_peak": float(np.mean(stats["autocorr_peak"])),
        }

    out_heatmaps = output_dir / "cross_layer_average_heatmaps.png"
    out_stats = output_dir / "cross_layer_temporal_stats.png"
    out_gallery = output_dir / "cross_layer_feature_gallery.png"

    save_average_heatmap_panel(layer_order, layer_heatmaps, out_heatmaps)
    save_temporal_stats_plot(layer_order, stats_summary, out_stats)
    feature_choices = save_feature_gallery(layer_order, layer_resampled_tracks, layer_stats, out_gallery)

    metadata = {
        "features_root": str(features_root),
        "sampled_track_count": len(sampled_rel_paths),
        "sampled_relative_paths": sampled_rel_paths,
        "layers": {layer_name: str(model_dir) for layer_name, model_dir in layer_specs},
        "stats_summary": stats_summary,
        "feature_choices": feature_choices,
    }
    metadata_path = output_dir / "cross_layer_report_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("Saved figures:")
    print(out_heatmaps)
    print(out_stats)
    print(out_gallery)
    print(metadata_path)


if __name__ == "__main__":
    main()
