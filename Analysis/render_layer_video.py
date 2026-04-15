import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Temporal-Music-Activations"))

from SparseAutoencoder import SparseAutoencoder

LAYER_NAMES = [f"layer_{i:02d}" for i in range(23)] + ["layer_final"]
SAE_ROOT = Path("/scratch/general/vast/u1406806/sae_output/models-all-layers-stride1-repacked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render synchronized cross-layer activation video with moving playhead."
    )
    parser.add_argument("--features_root", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--track_index", type=int, default=0)
    parser.add_argument("--video_bins", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--include_audio", action="store_true")
    return parser.parse_args()


def load_sae(layer_name: str, device: str) -> SparseAutoencoder | None:
    sae_dir = SAE_ROOT / layer_name
    checkpoint_candidates = [sae_dir / "sae_best.pt", sae_dir / "sae_final.pt"]
    ckpt_path = next((path for path in checkpoint_candidates if path.exists()), None)
    if ckpt_path is None:
        print(f"  [skip] {layer_name}: no checkpoint found in {sae_dir}", flush=True)
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    input_dim = state["encoder.weight"].shape[1]
    latent_dim = state["encoder.weight"].shape[0]

    model = SparseAutoencoder(input_dim, latent_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    print(
        f"  [ok]   {layer_name}: loaded {ckpt_path.name} "
        f"(input={input_dim}, latent={latent_dim})",
        flush=True,
    )
    return model


def collect_track_paths(features_root: Path, layer_name: str) -> List[Path]:
    return sorted((features_root / layer_name).rglob("*.npy"))


def encode_track(npy_path: Path, model: SparseAutoencoder, device: str) -> np.ndarray:
    features = np.load(npy_path)
    input_dim = model.encoder.weight.shape[1]
    if features.ndim != 2 or features.shape[1] != input_dim:
        raise ValueError(
            f"Unexpected feature shape for {npy_path}: {features.shape}, expected [T, {input_dim}]"
        )

    with torch.no_grad():
        tensor = torch.tensor(features, dtype=torch.float32, device=device)
        latent = model.activation(model.encoder(tensor))
    return latent.cpu().numpy()


def resample_layer(track_acts: np.ndarray, n_bins: int) -> np.ndarray:
    old_x = np.linspace(0, 1, track_acts.shape[0])
    new_x = np.linspace(0, 1, n_bins)
    return np.stack(
        [np.interp(new_x, old_x, track_acts[:, feat_idx]) for feat_idx in range(track_acts.shape[1])],
        axis=0,
    )


def probe_audio_duration(audio_path: Path | None) -> float | None:
    if audio_path is None or not audio_path.exists():
        return None
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def format_mmss(seconds: float) -> str:
    whole_seconds = max(0, int(seconds))
    minutes, remaining = divmod(whole_seconds, 60)
    return f"{minutes:02d}:{remaining:02d}"


def render_video(
    resampled_layers: np.ndarray,
    layer_names: List[str],
    output_path: Path,
    fps: int,
    dpi: int,
    audio_path: Path | None,
    include_audio: bool,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render MP4 output but was not found in PATH")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_output = output_path.with_name(f"{output_path.stem}_silent{output_path.suffix}")

    # resampled_layers shape: [n_layers, latent_dim, n_bins]
    n_layers, latent_dim, n_bins = resampled_layers.shape
    audio_duration = probe_audio_duration(audio_path)

    # Number of video frames is driven by the actual song duration so the video
    # plays in real time.  n_bins is just the heatmap's time resolution; each
    # frame maps to a bin via linear interpolation.
    if audio_duration is not None:
        n_frames = max(1, int(round(audio_duration * fps)))
    else:
        n_frames = n_bins
    print(
        f"Video: {n_frames} frames @ {fps} fps = {n_frames / fps:.1f}s "
        f"(heatmap bins: {n_bins})",
        flush=True,
    )

    vmax = float(np.quantile(resampled_layers, 0.995))
    vmax = max(vmax, 1e-6)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[18, 1], hspace=0.15)
    ax_heat = fig.add_subplot(gs[0])
    ax_prog = fig.add_subplot(gs[1])

    first_matrix = resampled_layers[:, :, 0].T  # [latent_dim, n_layers]
    image = ax_heat.imshow(
        first_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        origin="upper",
        vmin=0.0,
        vmax=vmax,
    )
    cbar = fig.colorbar(image, ax=ax_heat, pad=0.01)
    cbar.set_label("SAE activation", fontsize=10)

    ax_heat.set_title("New World Symphony | Synchronized Layer Activations", fontsize=14, pad=10)
    ax_heat.set_xlabel("Decoder layer", fontsize=11)
    ax_heat.set_ylabel("Latent feature index", fontsize=11)
    ax_heat.set_xticks(np.arange(n_layers))
    ax_heat.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
    y_ticks = np.linspace(0, latent_dim - 1, 9).astype(int)
    ax_heat.set_yticks(y_ticks)
    ax_heat.set_yticklabels([str(int(v)) for v in y_ticks], fontsize=8)

    ax_prog.set_xlim(0, n_bins - 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.set_yticks([])
    ax_prog.set_xticks(np.linspace(0, n_bins - 1, 6))
    ax_prog.set_xticklabels([f"{int(p)}%" for p in np.linspace(0, 100, 6)], fontsize=8)
    ax_prog.set_xlabel("Song timeline", fontsize=10)
    ax_prog.fill_between([0, n_bins - 1], 0, 1, color="#2f2f2f", alpha=0.2)
    playhead = ax_prog.axvline(0, color="red", linewidth=2.2, alpha=0.95)

    time_text = ax_heat.text(
        0.99,
        1.01,
        "",
        transform=ax_heat.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.35, "pad": 4, "edgecolor": "none"},
    )

    def update(frame_idx: int):
        frame_idx = int(frame_idx)
        # Map video frame → heatmap bin (time resolution ≠ frame count)
        bin_idx = int(round(frame_idx * (n_bins - 1) / max(n_frames - 1, 1)))
        bin_idx = min(bin_idx, n_bins - 1)

        matrix = resampled_layers[:, :, bin_idx].T
        image.set_data(matrix)
        playhead.set_xdata([bin_idx, bin_idx])

        progress = frame_idx / max(n_frames - 1, 1)
        if audio_duration is not None:
            current_seconds = progress * audio_duration
            label = f"Time: {format_mmss(current_seconds)} / {format_mmss(audio_duration)}"
        else:
            label = f"Progress: {progress * 100:5.1f}%"
        time_text.set_text(label)
        return [image, playhead, time_text]

    animation = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=7000, extra_args=["-pix_fmt", "yuv420p"])
    animation.save(str(silent_output), writer=writer, dpi=dpi)
    plt.close(fig)

    if include_audio and audio_path is not None and audio_path.exists():
        mux_result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(silent_output),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if mux_result.returncode == 0:
            silent_output.unlink(missing_ok=True)
            return
        print("[warn] ffmpeg audio mux failed; keeping silent video.", flush=True)
        print(mux_result.stderr, flush=True)

    silent_output.replace(output_path)


def main() -> None:
    args = parse_args()
    features_root = Path(args.features_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    audio_path = Path(args.audio_path).expanduser().resolve() if args.audio_path else None

    print(f"Repo root:  {REPO_ROOT}", flush=True)
    print(f"Features:   {features_root}", flush=True)
    print(f"Output:     {output_path}", flush=True)
    print(f"Audio:      {audio_path if audio_path else 'none'}", flush=True)
    print(f"Video bins: {args.video_bins}", flush=True)
    print(f"FPS:        {args.fps}", flush=True)

    collected_layers: List[np.ndarray] = []
    loaded_layer_names: List[str] = []
    track_stem = None

    print("Loading SAEs and encoding synchronized layer activations:", flush=True)
    for layer_name in LAYER_NAMES:
        model = load_sae(layer_name, args.device)
        if model is None:
            continue

        npy_files = collect_track_paths(features_root, layer_name)
        if not npy_files:
            print(f"  [skip] {layer_name}: no tracks found", flush=True)
            continue
        if args.track_index < 0 or args.track_index >= len(npy_files):
            raise IndexError(
                f"track_index={args.track_index} out of range for {layer_name} ({len(npy_files)} files)"
            )

        track_path = npy_files[args.track_index]
        if track_stem is None:
            track_stem = track_path.stem

        latent = encode_track(track_path, model, args.device)  # [T, latent_dim]
        resampled = resample_layer(latent, args.video_bins)    # [latent_dim, n_bins]
        collected_layers.append(resampled)
        loaded_layer_names.append(layer_name)
        print(f"  [ready] {layer_name}: latent {latent.shape} -> resampled {resampled.shape}", flush=True)

    if not collected_layers:
        raise RuntimeError("No layer activations could be prepared for rendering.")

    resampled_layers = np.stack(collected_layers, axis=0)  # [n_layers, latent_dim, n_bins]
    print(f"Stacked tensor for render: {resampled_layers.shape}", flush=True)

    render_video(
        resampled_layers=resampled_layers,
        layer_names=loaded_layer_names,
        output_path=output_path,
        fps=args.fps,
        dpi=args.dpi,
        audio_path=audio_path,
        include_audio=args.include_audio,
    )

    print(f"Saved synchronized video: {output_path}", flush=True)
    if track_stem:
        print(f"Track: {track_stem}", flush=True)


if __name__ == "__main__":
    main()
