import argparse
import json
import os
from datetime import datetime
from typing import List

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repack a selected staged feature subset into deterministic sharded .npy files."
    )
    parser.add_argument("--data_dir", required=True, help="Staged feature directory containing selected files.")
    parser.add_argument("--manifest_path", required=True, help="Relative-path manifest in selected-file order.")
    parser.add_argument("--output_dir", required=True, help="Output directory for repacked shards.")
    parser.add_argument(
        "--target_shard_size_mb",
        type=int,
        default=512,
        help="Target approximate shard size in MB before rolling to a new shard.",
    )
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="Optional explicit metadata path. Defaults to <output_dir>/repacked_metadata.json.",
    )
    return parser.parse_args()


def load_feature_array(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        array = np.load(path)
    else:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor in {path}, got {type(tensor)}")
        array = tensor.detach().cpu().numpy()

    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(-1, array.shape[-1])

    return np.asarray(array, dtype=np.float32)


def flush_shard(output_dir: str, shard_index: int, arrays: List[np.ndarray]) -> tuple[str, int]:
    shard_name = f"shard_{shard_index:05d}.npy"
    shard_path = os.path.join(output_dir, shard_name)
    shard_array = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
    np.save(shard_path, shard_array)
    return shard_name, int(shard_array.shape[0])


def main():
    args = parse_args()
    metadata_path = args.metadata_path or os.path.join(args.output_dir, "repacked_metadata.json")
    target_shard_size_bytes = max(1, int(args.target_shard_size_mb)) * 1024 * 1024

    os.makedirs(args.output_dir, exist_ok=True)

    relative_paths: List[str] = []
    with open(args.manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if entry:
                relative_paths.append(entry)

    if not relative_paths:
        raise ValueError(f"Manifest is empty: {args.manifest_path}")

    file_entries = []
    shard_arrays: List[np.ndarray] = []
    shard_bytes = 0
    shard_frame_cursor = 0
    shard_index = 0
    total_frames = 0
    input_dim = None
    written_shards = []

    for relative_path in relative_paths:
        source_path = os.path.join(args.data_dir, relative_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Manifest entry does not exist under data_dir: {source_path}")

        array = load_feature_array(source_path)
        if input_dim is None:
            input_dim = int(array.shape[-1])
        elif int(array.shape[-1]) != input_dim:
            raise ValueError(
                f"Input dimension mismatch for {source_path}: expected {input_dim}, got {array.shape[-1]}"
            )

        array_bytes = int(array.nbytes)
        if shard_arrays and shard_bytes + array_bytes > target_shard_size_bytes:
            shard_name, shard_frame_count = flush_shard(args.output_dir, shard_index, shard_arrays)
            written_shards.append({"shard_file": shard_name, "frame_count": shard_frame_count})
            shard_arrays = []
            shard_bytes = 0
            shard_frame_cursor = 0
            shard_index += 1

        shard_name = f"shard_{shard_index:05d}.npy"
        file_entries.append(
            {
                "relative_path": relative_path,
                "shard_file": shard_name,
                "shard_frame_start": shard_frame_cursor,
                "frame_count": int(array.shape[0]),
            }
        )
        shard_arrays.append(array)
        shard_bytes += array_bytes
        shard_frame_cursor += int(array.shape[0])
        total_frames += int(array.shape[0])

    if shard_arrays:
        shard_name, shard_frame_count = flush_shard(args.output_dir, shard_index, shard_arrays)
        written_shards.append({"shard_file": shard_name, "frame_count": shard_frame_count})

    metadata = {
        "format": "repacked_sharded_npy_v1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_data_dir": os.path.abspath(args.data_dir),
        "source_manifest_path": os.path.abspath(args.manifest_path),
        "output_dir": os.path.abspath(args.output_dir),
        "target_shard_size_mb": int(args.target_shard_size_mb),
        "selected_files": len(relative_paths),
        "total_frames": int(total_frames),
        "input_dim": int(input_dim),
        "shards": written_shards,
        "file_entries": file_entries,
    }

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"Repacked {len(relative_paths)} files into {len(written_shards)} shards "
        f"with {total_frames} total frames at {metadata_path}"
    )


if __name__ == "__main__":
    main()