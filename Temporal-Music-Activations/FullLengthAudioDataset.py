import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FullLengthAudioDataset(Dataset):
    """Dataset that loads precomputed feature tensors from .pt or .npy files.

    Expected layout:
    - data_dir can contain files directly or in subdirectories.
    - each sample file should contain either a 1D tensor [D] or 2D tensor [T, D].

    Sampling modes:
    - mean: each file is one sample; temporal tensors [T, D] are reduced to [D]
    - frames: each timestep in a temporal tensor becomes an individual sample [D]

    The default frame-based mode preserves temporal structure during SAE training,
    while still allowing explicit mean pooling when that is desired.
    """

    def __init__(
        self,
        data_dir: str,
        max_files: int = 0,
        random_subset_files: int = 0,
        file_manifest_path: Optional[str] = None,
        repacked_metadata_path: Optional[str] = None,
        subset_seed: int = 42,
        sample_mode: str = "frames",
        frame_stride: int = 1,
        max_frames: int = 0,
    ):
        self.data_dir = data_dir
        self.max_files = max(0, int(max_files))
        self.random_subset_files = max(0, int(random_subset_files))
        self.file_manifest_path = file_manifest_path
        self.repacked_metadata_path = repacked_metadata_path
        self.subset_seed = int(subset_seed)
        self.sample_mode = sample_mode
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames = max(0, int(max_frames))
        self._cached_tensor_path: Optional[str] = None
        self._cached_tensor: Optional[torch.Tensor] = None
        self._cached_frame_array_path: Optional[str] = None
        self._cached_frame_array = None
        self.data_format = "individual_files"
        self.repacked_metadata: Optional[Dict] = None
        self._repacked_file_entries: List[Dict] = []

        auto_repacked_metadata_path = self.repacked_metadata_path
        if auto_repacked_metadata_path is None:
            candidate_metadata_path = os.path.join(data_dir, "repacked_metadata.json")
            if os.path.isfile(candidate_metadata_path):
                auto_repacked_metadata_path = candidate_metadata_path

        if auto_repacked_metadata_path is not None:
            self.repacked_metadata_path = auto_repacked_metadata_path
            self._load_repacked_metadata(auto_repacked_metadata_path)
            self.files = [entry["relative_path"] for entry in self._repacked_file_entries]
            self.total_files_discovered = int(self.repacked_metadata["selected_files"])
        else:
            self.files = self._resolve_files(data_dir)
            self.total_files_discovered = len(self.files)
            if self.file_manifest_path is None:
                self.files = self._apply_file_selection(self.files)
        self.selected_files = len(self.files)
        if not self.files:
            raise ValueError(
                f"No .pt or .npy files found under: {data_dir}. "
                "Place precomputed feature tensors there before training."
            )

        if self.sample_mode not in {"mean", "frames"}:
            raise ValueError(
                f"Unsupported sample_mode={self.sample_mode!r}. Use 'mean' or 'frames'."
            )

        if self.data_format == "repacked_shards":
            self.input_dim = int(self.repacked_metadata["input_dim"])
        else:
            sample = self._load_tensor(self.files[0])
            self.input_dim = int(sample.shape[-1])
        self.frame_index: List[Tuple[int, int]] = []
        self.file_frame_ranges: List[Tuple[int, int]] = []
        if self.sample_mode == "frames":
            self.frame_index = self._build_frame_index()
            if not self.frame_index:
                raise ValueError(
                    f"No frame samples found under: {data_dir}. "
                    "Ensure tensors are 1D or 2D with a non-zero time dimension."
                )

    def _collect_files(self, root: str) -> List[str]:
        files: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                lower = name.lower()
                if lower.endswith(".pt") or lower.endswith(".npy"):
                    files.append(os.path.join(dirpath, name))
        files.sort()
        return files

    def _resolve_files(self, root: str) -> List[str]:
        if self.file_manifest_path is None:
            return self._collect_files(root)

        files: List[str] = []
        with open(self.file_manifest_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                entry = line.strip()
                if not entry:
                    continue
                path = entry if os.path.isabs(entry) else os.path.join(root, entry)
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"Manifest entry {entry!r} does not exist under data_dir={root}: {path}"
                    )
                files.append(path)

        if not files:
            raise ValueError(
                f"No feature files were loaded from manifest: {self.file_manifest_path}"
            )
        return files

    def _load_repacked_metadata(self, metadata_path: str) -> None:
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)

        if metadata.get("format") != "repacked_sharded_npy_v1":
            raise ValueError(
                f"Unsupported repacked metadata format in {metadata_path}: {metadata.get('format')!r}"
            )

        file_entries = metadata.get("file_entries", [])
        if not file_entries:
            raise ValueError(f"Repacked metadata contains no file entries: {metadata_path}")

        for entry in file_entries:
            required_keys = {"relative_path", "shard_file", "shard_frame_start", "frame_count"}
            missing_keys = required_keys.difference(entry)
            if missing_keys:
                raise ValueError(
                    f"Repacked metadata entry is missing keys {sorted(missing_keys)}: {entry}"
                )
            shard_path = os.path.join(self.data_dir, entry["shard_file"])
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(
                    f"Repacked shard file listed in metadata does not exist: {shard_path}"
                )

        self.repacked_metadata = metadata
        self._repacked_file_entries = file_entries
        self.data_format = "repacked_shards"

    def _apply_file_selection(self, files: List[str]) -> List[str]:
        if self.max_files > 0 and self.random_subset_files > 0:
            raise ValueError(
                "max_files and random_subset_files are mutually exclusive. "
                "Use max_files for quick ordered debugging runs or random_subset_files "
                "for reproducible reduced-corpus training."
            )

        selected_files = files
        if self.max_files > 0:
            selected_files = selected_files[:self.max_files]

        if self.random_subset_files > 0:
            if self.random_subset_files > len(selected_files):
                raise ValueError(
                    f"random_subset_files={self.random_subset_files} exceeds the number of available files "
                    f"({len(selected_files)}) under {self.data_dir}."
                )
            rng = np.random.default_rng(self.subset_seed)
            subset_indices = np.sort(
                rng.choice(len(selected_files), size=self.random_subset_files, replace=False)
            )
            selected_files = [selected_files[idx] for idx in subset_indices]

        return selected_files

    def _load_tensor(self, path: str, cache: bool = True) -> torch.Tensor:
        if cache and path == self._cached_tensor_path and self._cached_tensor is not None:
            return self._cached_tensor

        if path.lower().endswith(".pt"):
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor in {path}, got {type(tensor)}")
        else:
            array = np.load(path)
            tensor = torch.from_numpy(array)

        tensor = tensor.float()
        tensor = self._normalize_temporal_tensor(tensor)

        if cache:
            self._cached_tensor_path = path
            self._cached_tensor = tensor

        return tensor

    def _load_frame_array(self, path: str):
        if path == self._cached_frame_array_path and self._cached_frame_array is not None:
            return self._cached_frame_array

        array = np.load(path, mmap_mode="r")
        if array.ndim > 2:
            array = array.reshape(-1, array.shape[-1])

        self._cached_frame_array_path = path
        self._cached_frame_array = array
        return array

    def _normalize_temporal_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert supported inputs to either [D] or [T, D]."""
        if tensor.ndim == 1:
            return tensor
        if tensor.ndim == 2:
            return tensor
        return tensor.reshape(-1, tensor.shape[-1])

    def _build_frame_index(self) -> List[Tuple[int, int]]:
        """Map dataset indices to (file_index, frame_index) pairs.

        This lets the SAE train directly on timestep vectors instead of averaging
        away the temporal dimension before the model sees the data.
        """
        frame_index: List[Tuple[int, int]] = []
        self.file_frame_ranges = []
        for file_idx, path in enumerate(self.files):
            file_start = len(frame_index)
            num_frames = self._get_num_frames(path)
            if num_frames == 1:
                frame_index.append((file_idx, 0))
                self.file_frame_ranges.append((file_start, len(frame_index)))
                continue
            for frame_idx in range(0, num_frames, self.frame_stride):
                frame_index.append((file_idx, frame_idx))
                if self.max_frames and len(frame_index) >= self.max_frames:
                    self.file_frame_ranges.append((file_start, len(frame_index)))
                    return frame_index
            self.file_frame_ranges.append((file_start, len(frame_index)))
        return frame_index

    def get_file_frame_ranges(self) -> List[Tuple[int, int]]:
        """Return [start, end) frame-index spans for each selected file."""
        if self.sample_mode != "frames":
            raise ValueError("File frame ranges are only available when sample_mode='frames'.")
        return list(self.file_frame_ranges)

    def get_selected_relative_files(self) -> List[str]:
        """Return selected files as paths relative to data_dir when possible."""
        if self.data_format == "repacked_shards":
            return [entry["relative_path"] for entry in self._repacked_file_entries]

        relative_files: List[str] = []
        for path in self.files:
            if os.path.isabs(path):
                relative_files.append(os.path.relpath(path, self.data_dir))
            else:
                relative_files.append(path)
        return relative_files

    def _get_num_frames(self, path: str) -> int:
        """Return the number of timestep frames in a feature file.

        For .npy files, uses a memory-mapped open so only the array header is
        read from disk — the data pages are never faulted in.  This makes index
        construction O(num_files) in metadata I/O rather than O(total_bytes).
        For .pt files a full load is unavoidable (PyTorch has no header-only API).
        """
        if self.data_format == "repacked_shards":
            file_idx = self.files.index(path)
            return int(self._repacked_file_entries[file_idx]["frame_count"])

        if path.lower().endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            return 1 if arr.ndim == 1 else int(arr.shape[0])
        # .pt path: load once and discard data immediately
        tensor = self._load_tensor(path, cache=False)
        return 1 if tensor.ndim == 1 else int(tensor.shape[0])

    def _get_frame_sample(self, file_idx: int, frame_idx: int) -> torch.Tensor:
        if self.data_format == "repacked_shards":
            entry = self._repacked_file_entries[file_idx]
            shard_path = os.path.join(self.data_dir, entry["shard_file"])
            array = self._load_frame_array(shard_path)
            shard_frame_idx = int(entry["shard_frame_start"]) + int(frame_idx)
            if shard_frame_idx >= int(entry["shard_frame_start"]) + int(entry["frame_count"]):
                raise IndexError(
                    f"Frame index {frame_idx} out of range for repacked file {entry['relative_path']} "
                    f"with {entry['frame_count']} frames"
                )
            return torch.from_numpy(np.array(array[shard_frame_idx], dtype=np.float32, copy=True))

        path = self.files[file_idx]

        if path.lower().endswith(".npy"):
            array = self._load_frame_array(path)
            if array.ndim == 1:
                return torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
            if frame_idx >= array.shape[0]:
                raise IndexError(
                    f"Frame index {frame_idx} out of range for {path} "
                    f"with {array.shape[0]} frames"
                )
            return torch.from_numpy(np.array(array[frame_idx], dtype=np.float32, copy=True))

        tensor = self._load_tensor(path)
        if tensor.ndim == 1:
            return tensor
        if frame_idx >= tensor.shape[0]:
            raise IndexError(
                f"Frame index {frame_idx} out of range for {path} "
                f"with {tensor.shape[0]} frames"
            )
        return tensor[frame_idx]

    def __len__(self) -> int:
        if self.sample_mode == "frames":
            return len(self.frame_index)
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.sample_mode == "frames":
            file_idx, frame_idx = self.frame_index[idx]
            tensor = self._get_frame_sample(file_idx, frame_idx)
        else:
            if self.data_format == "repacked_shards":
                entry = self._repacked_file_entries[idx]
                shard_path = os.path.join(self.data_dir, entry["shard_file"])
                array = self._load_frame_array(shard_path)
                start = int(entry["shard_frame_start"])
                end = start + int(entry["frame_count"])
                if int(entry["frame_count"]) == 1:
                    tensor = torch.from_numpy(np.array(array[start], dtype=np.float32, copy=True))
                else:
                    tensor = torch.from_numpy(np.array(array[start:end], dtype=np.float32, copy=True)).mean(dim=0)
            else:
                tensor = self._normalize_temporal_tensor(self._load_tensor(self.files[idx]))
                if tensor.ndim == 2:
                    tensor = tensor.mean(dim=0)

        if tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature size mismatch for sample index {idx}: "
                f"expected {self.input_dim}, got {tensor.shape[-1]}"
            )
        return tensor
