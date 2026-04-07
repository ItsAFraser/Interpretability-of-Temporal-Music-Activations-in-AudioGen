"""Extract per-timestep residual activations from a MusicGen decoder layer.

This script turns audio files into feature tensors that the SAE can train on.
Unlike the original mean-pooled workflow, it saves the full temporal sequence by
default so downstream analysis can preserve song structure.

Extraction recipe:
1. Load audio and resample to MusicGen's expected sample rate.
2. Encode the waveform into EnCodec audio tokens.
3. Run those tokens through MusicGen's decoder with unconditional text context.
4. Save one feature tensor per track, usually with shape [timesteps, hidden_dim].
"""

import argparse
import json
from pathlib import Path
import time
from typing import Any, List, Sequence

import numpy as np
import torch


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MusicGen decoder residual features from audio files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing audio files to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where feature tensors will be written.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/musicgen-small",
        help="Hugging Face MusicGen checkpoint name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--decoder_layer",
        type=int,
        default=-1,
        help="Single decoder layer index. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--decoder_layers",
        type=str,
        default="",
        help=(
            "Optional comma-separated decoder layer list, e.g. '0,8,16,-1'. "
            "When provided, extraction runs once per track and saves each selected layer."
        ),
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="none",
        choices=["none", "mean", "max"],
        help="Optional pooling applied before saving. 'none' preserves time.",
    )
    parser.add_argument(
        "--audio_glob",
        type=str,
        default="**/*",
        help="Glob pattern, relative to input_dir, used to find audio files.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help=(
            "Optional text file containing one audio path per line. Relative paths "
            "are resolved from --input_dir. When provided, this avoids a full "
            "filesystem scan on every extraction task."
        ),
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Optional cap on the number of audio files to process.",
    )
    parser.add_argument(
        "--file_start",
        type=int,
        default=0,
        help="Zero-based index of the first sorted audio file to process.",
    )
    parser.add_argument(
        "--file_count",
        type=int,
        default=0,
        help="Optional number of sorted audio files to process starting at --file_start.",
    )
    parser.add_argument(
        "--max_duration_sec",
        type=float,
        default=0.0,
        help="Optional max duration per track after loading and resampling.",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="npy",
        choices=["npy", "pt"],
        help="On-disk format for extracted features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite feature files that already exist.",
    )
    parser.add_argument(
        "--metadata_json",
        action="store_true",
        help="Write a sidecar JSON file with extraction metadata for each track.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_audio_files(
    input_dir: Path,
    audio_glob: str,
    max_files: int,
    manifest_path: str,
) -> List[Path]:
    if manifest_path:
        manifest_file = Path(manifest_path).expanduser().resolve()
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {manifest_file}")

        paths = []
        for line in manifest_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            path = Path(line)
            if not path.is_absolute():
                path = input_dir / path
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                paths.append(path)
    else:
        paths = [
            path for path in input_dir.glob(audio_glob)
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ]
        paths.sort()

    if max_files and max_files > 0:
        paths = paths[:max_files]
    return paths


def select_audio_slice(audio_paths: List[Path], file_start: int, file_count: int) -> List[Path]:
    """Select a deterministic slice of the sorted audio file list."""
    if file_start < 0:
        raise ValueError(f"file_start must be >= 0, got {file_start}")
    if file_count < 0:
        raise ValueError(f"file_count must be >= 0, got {file_count}")
    if file_start >= len(audio_paths):
        return []

    if file_count == 0:
        return audio_paths[file_start:]
    return audio_paths[file_start:file_start + file_count]


def load_audio(path: Path, sample_rate: int, max_duration_sec: float) -> np.ndarray:
    import librosa

    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    if max_duration_sec and max_duration_sec > 0:
        max_samples = int(sample_rate * max_duration_sec)
        waveform = waveform[:max_samples]
    return waveform.astype(np.float32, copy=False)


def flatten_audio_codes(audio_codes: torch.LongTensor) -> torch.Tensor:
    """Convert EnCodec codes to [batch*codebooks, total_steps] for the MusicGen decoder.

    Handles two layouts emitted by different transformers versions:
    - 3-D [batch, num_codebooks, seq_len]   — transformers >= ~4.40 (current)
    - 4-D [frames, batch, codebooks, steps] — legacy layout
    """
    if audio_codes.ndim == 3:
        # Current transformers: audio_codes shape is [batch, num_codebooks, seq_len]
        batch_size, num_codebooks, seq_len = audio_codes.shape
        return audio_codes.reshape(batch_size * num_codebooks, seq_len)
    elif audio_codes.ndim == 4:
        # Legacy layout: [frames, batch, codebooks, steps]
        num_frames, batch_size, num_codebooks, frame_length = audio_codes.shape
        return audio_codes.permute(1, 2, 0, 3).reshape(
            batch_size * num_codebooks, num_frames * frame_length
        )
    else:
        raise ValueError(
            f"Expected audio_codes with 3 or 4 dims, got shape {tuple(audio_codes.shape)}"
        )


def get_unconditional_context(
    model: Any,
    processor: Any,
    device: str,
) -> dict:
    """Build the null text conditioning used for decoder-only audio analysis."""
    if hasattr(model, "get_unconditional_inputs"):
        unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
    else:
        unconditional_inputs = processor(text=[""], padding=True, return_tensors="pt")

    if hasattr(unconditional_inputs, "_asdict"):
        unconditional_inputs = unconditional_inputs._asdict()
    elif not isinstance(unconditional_inputs, dict):
        unconditional_inputs = dict(unconditional_inputs)

    def move_to_device(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(move_to_device(item) for item in value)
        if isinstance(value, list):
            return [move_to_device(item) for item in value]
        if isinstance(value, dict):
            return {key: move_to_device(item) for key, item in value.items()}
        return value

    moved_inputs = {key: move_to_device(value) for key, value in unconditional_inputs.items()}
    if "encoder_outputs" in moved_inputs:
        return {
            "encoder_outputs": moved_inputs["encoder_outputs"],
            "attention_mask": moved_inputs.get("attention_mask"),
        }

    return {
        key: moved_inputs[key]
        for key in ("input_ids", "attention_mask")
        if key in moved_inputs
    }


def select_decoder_hidden_state(
    decoder_hidden_states: Sequence[torch.Tensor],
    decoder_layer: int,
) -> torch.Tensor:
    """Map a transformer-layer index onto the hidden-states tuple.

    Hugging Face returns one tensor for the embedding output followed by one
    tensor per decoder block. A decoder_layer of 0 selects the output after the
    first transformer block. A value of -1 selects the final decoder output.
    """
    num_block_outputs = len(decoder_hidden_states) - 1
    if num_block_outputs <= 0:
        raise ValueError("MusicGen did not return decoder block hidden states.")

    if decoder_layer == -1:
        hidden_state_index = len(decoder_hidden_states) - 1
    else:
        if decoder_layer < 0 or decoder_layer >= num_block_outputs:
            raise ValueError(
                f"decoder_layer must be in [0, {num_block_outputs - 1}] or -1, got {decoder_layer}"
            )
        hidden_state_index = decoder_layer + 1
    return decoder_hidden_states[hidden_state_index]


def parse_requested_layers(decoder_layer: int, decoder_layers_arg: str) -> List[int]:
    """Parse and normalize requested decoder layers.

    We support both the original --decoder_layer flag and the new --decoder_layers
    list flag. If the list flag is set, it takes precedence.
    """
    if decoder_layers_arg.strip():
        layers: List[int] = []
        for token in decoder_layers_arg.split(","):
            token = token.strip()
            if not token:
                continue
            layers.append(int(token))
    else:
        layers = [int(decoder_layer)]

    # Preserve order while removing duplicates to avoid redundant writes.
    seen = set()
    normalized_layers: List[int] = []
    for layer in layers:
        if layer in seen:
            continue
        seen.add(layer)
        normalized_layers.append(layer)

    if not normalized_layers:
        raise ValueError("No decoder layers requested. Use --decoder_layer or --decoder_layers.")
    return normalized_layers


def get_layer_subdir_name(layer: int) -> str:
    """Map layer ids to deterministic output subdirectories."""
    if layer == -1:
        return "layer_final"
    return f"layer_{layer:02d}"


def apply_pooling(features: torch.Tensor, pooling: str) -> torch.Tensor:
    if features.ndim == 1:
        return features
    if pooling == "none":
        return features
    if pooling == "mean":
        return features.mean(dim=0)
    if pooling == "max":
        return features.max(dim=0).values
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def save_tensor(output_path: Path, tensor: torch.Tensor, save_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_format == "pt":
        torch.save(tensor.cpu(), output_path)
        return
    np.save(output_path, tensor.cpu().numpy())


def write_metadata(
    metadata_path: Path,
    source_path: Path,
    model_name: str,
    sample_rate: int,
    decoder_layer: int,
    requested_decoder_layers: Sequence[int],
    pooling: str,
    feature_shape: Sequence[int],
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_path": str(source_path),
        "model_name": model_name,
        "sample_rate": sample_rate,
        "decoder_layer": decoder_layer,
        "requested_decoder_layers": list(requested_decoder_layers),
        "pooling": pooling,
        "feature_shape": list(feature_shape),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def format_elapsed_seconds(elapsed_seconds: float) -> str:
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def main() -> None:
    args = parse_args()
    from tqdm import tqdm
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    device = resolve_device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_name)
    # `from_pretrained` type stubs can be imprecise across transformers versions,
    # so we keep this as Any for static checkers while preserving runtime behavior.
    model: Any = MusicgenForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    print(
        f"Loaded {args.model_name} on {device}; preparing extraction inputs.",
        flush=True,
    )

    sample_rate = int(model.config.audio_encoder.sampling_rate)
    requested_layers = parse_requested_layers(args.decoder_layer, args.decoder_layers)
    unconditional_context = get_unconditional_context(model, processor, device)
    audio_paths = collect_audio_files(
        input_dir,
        args.audio_glob,
        args.max_files,
        args.manifest_path,
    )
    if not audio_paths:
        raise ValueError(
            f"No audio files found under {input_dir} with glob {args.audio_glob!r}"
        )
    audio_paths = select_audio_slice(audio_paths, args.file_start, args.file_count)
    if not audio_paths:
        print(
            "The requested file slice is empty; nothing to do for "
            f"file_start={args.file_start} and file_count={args.file_count}.",
            flush=True,
        )
        return

    print(
        f"Extracting MusicGen features from {len(audio_paths)} files "
        f"using {args.model_name} on {device} for layers {requested_layers} "
        f"(file_start={args.file_start}, file_count={args.file_count or 'all'})",
        flush=True,
    )

    overall_start_time = time.perf_counter()
    for file_index, audio_path in enumerate(tqdm(audio_paths, desc="Extracting", unit="file"), start=1):
        file_start_time = time.perf_counter()
        relative_path = audio_path.relative_to(input_dir)
        print(
            f"[{file_index}/{len(audio_paths)}] Starting {relative_path}",
            flush=True,
        )
        layer_targets = []
        for layer in requested_layers:
            layer_subdir = get_layer_subdir_name(layer)
            output_stem = output_dir / layer_subdir / relative_path.with_suffix("")
            feature_path = output_stem.with_suffix(f".{args.save_format}")
            metadata_path = output_stem.with_suffix(".json")
            layer_targets.append((layer, feature_path, metadata_path))

        # If every target already exists and overwrite is disabled, skip this track
        # before doing model inference to avoid unnecessary compute.
        if not args.overwrite and all(target[1].exists() for target in layer_targets):
            print(
                f"[{file_index}/{len(audio_paths)}] Skipping {relative_path} because all outputs already exist.",
                flush=True,
            )
            continue

        waveform = load_audio(audio_path, sample_rate, args.max_duration_sec)
        if waveform.size == 0:
            raise ValueError(f"Audio file is empty after loading: {audio_path}")

        audio_inputs = processor(
            audio=waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        audio_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in audio_inputs.items()
        }

        with torch.no_grad():
            encoder_outputs = model.audio_encoder.encode(
                audio_inputs["input_values"],
                audio_inputs.get("padding_mask"),
            )
            decoder_input_ids = flatten_audio_codes(encoder_outputs.audio_codes)
            outputs = model(
                **unconditional_context,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        # Reuse the same forward pass for all requested layers. This makes
        # multi-layer studies much faster than rerunning extraction per layer.
        for layer, feature_path, metadata_path in layer_targets:
            if feature_path.exists() and not args.overwrite:
                continue

            decoder_features = select_decoder_hidden_state(
                outputs.decoder_hidden_states,
                layer,
            )
            decoder_features = decoder_features.squeeze(0).detach().cpu()
            if decoder_features.ndim not in (1, 2):
                raise ValueError(
                    f"Unexpected decoder feature shape {tuple(decoder_features.shape)} "
                    f"for {audio_path} layer {layer}. Expected 1-D or 2-D [T, hidden_dim]."
                )
            decoder_features = apply_pooling(decoder_features, args.pooling)

            save_tensor(feature_path, decoder_features, args.save_format)
            if args.metadata_json:
                write_metadata(
                    metadata_path,
                    source_path=audio_path,
                    model_name=args.model_name,
                    sample_rate=sample_rate,
                    decoder_layer=layer,
                    requested_decoder_layers=requested_layers,
                    pooling=args.pooling,
                    feature_shape=tuple(decoder_features.shape),
                )

        file_elapsed = time.perf_counter() - file_start_time
        total_elapsed = time.perf_counter() - overall_start_time
        print(
            f"[{file_index}/{len(audio_paths)}] Finished {relative_path} "
            f"in {format_elapsed_seconds(file_elapsed)} "
            f"(total {format_elapsed_seconds(total_elapsed)})",
            flush=True,
        )

    print(f"Saved extracted features under {output_dir}", flush=True)


if __name__ == "__main__":
    main()