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
        help="0-based decoder transformer layer index. Use -1 for the final layer.",
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
        "--max_files",
        type=int,
        default=0,
        help="Optional cap on the number of audio files to process.",
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


def collect_audio_files(input_dir: Path, audio_glob: str, max_files: int) -> List[Path]:
    paths = [
        path for path in input_dir.glob(audio_glob)
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    paths.sort()
    if max_files and max_files > 0:
        paths = paths[:max_files]
    return paths


def load_audio(path: Path, sample_rate: int, max_duration_sec: float) -> np.ndarray:
    import librosa

    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    if max_duration_sec and max_duration_sec > 0:
        max_samples = int(sample_rate * max_duration_sec)
        waveform = waveform[:max_samples]
    return waveform.astype(np.float32, copy=False)


def flatten_audio_codes(audio_codes: torch.LongTensor) -> torch.LongTensor:
    """Convert EnCodec codes from [frames, batch, codebooks, steps] to [batch*codebooks, steps]."""
    if audio_codes.ndim != 4:
        raise ValueError(f"Expected audio_codes with 4 dims, got shape {tuple(audio_codes.shape)}")

    num_frames, batch_size, num_codebooks, frame_length = audio_codes.shape
    flattened = audio_codes.permute(1, 2, 0, 3).reshape(batch_size * num_codebooks, num_frames * frame_length)
    return flattened


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
    pooling: str,
    feature_shape: Sequence[int],
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_path": str(source_path),
        "model_name": model_name,
        "sample_rate": sample_rate,
        "decoder_layer": decoder_layer,
        "pooling": pooling,
        "feature_shape": list(feature_shape),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


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
    model = MusicgenForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    sample_rate = int(model.config.audio_encoder.sampling_rate)
    unconditional_context = get_unconditional_context(model, processor, device)
    audio_paths = collect_audio_files(input_dir, args.audio_glob, args.max_files)
    if not audio_paths:
        raise ValueError(
            f"No audio files found under {input_dir} with glob {args.audio_glob!r}"
        )

    print(
        f"Extracting MusicGen features from {len(audio_paths)} files "
        f"using {args.model_name} on {device}"
    )

    for audio_path in tqdm(audio_paths, desc="Extracting", unit="file"):
        relative_path = audio_path.relative_to(input_dir)
        output_stem = output_dir / relative_path.with_suffix("")
        feature_path = output_stem.with_suffix(f".{args.save_format}")
        metadata_path = output_stem.with_suffix(".json")

        if feature_path.exists() and not args.overwrite:
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

        decoder_features = select_decoder_hidden_state(
            outputs.decoder_hidden_states,
            args.decoder_layer,
        )
        decoder_features = decoder_features.squeeze(0).detach().cpu()
        decoder_features = apply_pooling(decoder_features, args.pooling)

        save_tensor(feature_path, decoder_features, args.save_format)
        if args.metadata_json:
            write_metadata(
                metadata_path,
                source_path=audio_path,
                model_name=args.model_name,
                sample_rate=sample_rate,
                decoder_layer=args.decoder_layer,
                pooling=args.pooling,
                feature_shape=tuple(decoder_features.shape),
            )

    print(f"Saved extracted features under {output_dir}")


if __name__ == "__main__":
    main()