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
            "Create one poster-ready figure showing representative temporal "
            "features across several SAE layers."
        )
    )
    parser.add_argument("--features_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--layer_spec",
        action="append",
        required=True,
        help="Layer-to-SAE mapping in the form layer_name=/path/to/layer_model_dir.",
    )
    parser.add_argument("--avg_tracks", type=int, default=100)
    parser.add_argument("--time_bins", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
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


def autocorr_peak(series: np.ndarray) -> float:
    if series.std() < 1e-8:
        return 0.0
    s = series - series.mean()
    full = np.correlate(s, s, mode="full")
    mid = len(full) // 2
    max_lag = min(len(series) // 2, 100)
    ac = full[mid:mid + max_lag + 1] / (full[mid] + 1e-12)
    return float(ac[1:].max()) if len(ac) > 1 else 0.0


def compute_feature_stats(resampled_tracks: list[np.ndarray]) -> dict[str, np.ndarray]:
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


def score_structural(stats: dict[str, np.ndarray]) -> np.ndarray:
    ac = stats["autocorr_peak"]
    act = stats["active_frac"]
    var = stats["temporal_var"]
    return ac * (0.5 + act) * (0.5 + np.sqrt(var + 1e-12))


def score_narrative(stats: dict[str, np.ndarray]) -> np.ndarray:
    trend = stats["trend_slope"]
    act = stats["active_frac"]
    var = stats["temporal_var"]
    return trend * (0.5 + act) * (0.5 + np.sqrt(var + 1e-12))


def select_feature(score: np.ndarray, stats: dict[str, np.ndarray], exclude: set[int] | None = None) -> int:
    exclude = exclude or set()
    order = np.argsort(score)[::-1]
    for idx in order:
        idx_int = int(idx)
        if idx_int in exclude:
            continue
        if stats["active_frac"][idx_int] < 0.01:
            continue
        return idx_int
    raise ValueError("Unable to select a representative feature.")


def normalize_series(series: np.ndarray) -> np.ndarray:
    return (series - series.min()) / (series.max() - series.min() + 1e-12)


def save_poster_figure(
    layer_order: list[str],
    layer_tracks: dict[str, list[np.ndarray]],
    feature_choices: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(layer_order), 2, figsize=(12, 3.0 * len(layer_order)), sharex=True)
    if len(layer_order) == 1:
        axes = np.array([axes])

    x_axis = np.linspace(0, 1, layer_tracks[layer_order[0]][0].shape[0])
    column_info = [
        ("structural_feature", "Structural-like", "#DD8452"),
        ("narrative_feature", "Narrative-like", "#55A868"),
    ]

    for row, layer_name in enumerate(layer_order):
        stacked = np.stack(layer_tracks[layer_name], axis=0)
        for col, (feature_key, title, color) in enumerate(column_info):
            feature_idx = feature_choices[layer_name][feature_key]
            ts = stacked[:, :, feature_idx].mean(axis=0)
            ts = normalize_series(ts)

            ax = axes[row, col]
            ax.plot(x_axis, ts, color=color, linewidth=1.8)
            ax.fill_between(x_axis, ts, alpha=0.15, color=color)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.15, linewidth=0.5)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(layer_name, fontsize=10, fontweight="bold")
            ax.text(
                0.98,
                0.92,
                f"f{feature_idx}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    axes[-1, 0].set_xlabel("Relative time")
    axes[-1, 1].set_xlabel("Relative time")
    fig.suptitle("Representative Temporal Features Across Layers", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close()


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
        rel_map = collect_relative_feature_map(layer_dir)
        if not rel_map:
            raise FileNotFoundError(f"No .npy files found in {layer_dir}")
        layer_rel_maps[layer_name] = rel_map

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

    layer_tracks: dict[str, list[np.ndarray]] = {}
    layer_feature_choices: dict[str, dict[str, int]] = {}

    for layer_name, model_dir in layer_specs:
        model = load_sae(model_dir)
        resampled_tracks: list[np.ndarray] = []
        for rel_path in sampled_rel_paths:
            npy_path = layer_rel_maps[layer_name][rel_path]
            acts = encode_track(model, npy_path)
            resampled_tracks.append(resample_feature_matrix(acts, args.time_bins))

        stats = compute_feature_stats(resampled_tracks)
        structural_idx = select_feature(score_structural(stats), stats)
        narrative_idx = select_feature(score_narrative(stats), stats, exclude={structural_idx})

        layer_tracks[layer_name] = resampled_tracks
        layer_feature_choices[layer_name] = {
            "structural_feature": structural_idx,
            "narrative_feature": narrative_idx,
        }

    out_path = output_dir / "poster_temporal_feature_figure.png"
    save_poster_figure(layer_order, layer_tracks, layer_feature_choices, out_path)

    manifest = {
        "features_root": str(features_root),
        "sampled_track_count": len(sampled_rel_paths),
        "sampled_relative_paths": sampled_rel_paths,
        "layers": {layer_name: str(model_dir) for layer_name, model_dir in layer_specs},
        "feature_choices": layer_feature_choices,
    }
    manifest_path = output_dir / "poster_temporal_feature_figure.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Saved:")
    print(out_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
