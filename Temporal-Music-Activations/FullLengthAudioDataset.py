import os
from typing import List, Sequence, Tuple

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
        sample_mode: str = "frames",
        frame_stride: int = 1,
        max_frames: int = 0,
    ):
        self.data_dir = data_dir
        self.sample_mode = sample_mode
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames = max(0, int(max_frames))
        self.files = self._collect_files(data_dir)
        if max_files and max_files > 0:
            self.files = self.files[:max_files]
        if not self.files:
            raise ValueError(
                f"No .pt or .npy files found under: {data_dir}. "
                "Place precomputed feature tensors there before training."
            )

        if self.sample_mode not in {"mean", "frames"}:
            raise ValueError(
                f"Unsupported sample_mode={self.sample_mode!r}. Use 'mean' or 'frames'."
            )

        sample = self._load_tensor(self.files[0])
        self.input_dim = int(sample.shape[-1])
        self.frame_index: List[Tuple[int, int]] = []
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

    def _load_tensor(self, path: str) -> torch.Tensor:
        if path.lower().endswith(".pt"):
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor in {path}, got {type(tensor)}")
        else:
            array = np.load(path)
            tensor = torch.from_numpy(array)

        tensor = tensor.float()
        if tensor.ndim == 1:
            return tensor
        if tensor.ndim == 2:
            return tensor

        # Collapse unexpected higher-rank tensors into a temporal matrix [T, D].
        return tensor.reshape(-1, tensor.shape[-1])

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
        for file_idx, path in enumerate(self.files):
            num_frames = self._get_num_frames(path)
            if num_frames == 1:
                frame_index.append((file_idx, 0))
                continue
            for frame_idx in range(0, num_frames, self.frame_stride):
                frame_index.append((file_idx, frame_idx))
                if self.max_frames and len(frame_index) >= self.max_frames:
                    return frame_index
        return frame_index

    def _get_num_frames(self, path: str) -> int:
        """Return the number of timestep frames in a feature file.

        For .npy files, uses a memory-mapped open so only the array header is
        read from disk — the data pages are never faulted in.  This makes index
        construction O(num_files) in metadata I/O rather than O(total_bytes).
        For .pt files a full load is unavoidable (PyTorch has no header-only API).
        """
        if path.lower().endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            return 1 if arr.ndim == 1 else int(arr.shape[0])
        # .pt path: load once and discard data immediately
        tensor = self._normalize_temporal_tensor(self._load_tensor(path))
        return 1 if tensor.ndim == 1 else int(tensor.shape[0])

    def _get_frame_sample(self, file_idx: int, frame_idx: int) -> torch.Tensor:
        tensor = self._normalize_temporal_tensor(self._load_tensor(self.files[file_idx]))
        if tensor.ndim == 1:
            return tensor
        if frame_idx >= tensor.shape[0]:
            raise IndexError(
                f"Frame index {frame_idx} out of range for {self.files[file_idx]} "
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
            tensor = self._normalize_temporal_tensor(self._load_tensor(self.files[idx]))
            if tensor.ndim == 2:
                tensor = tensor.mean(dim=0)

        if tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature size mismatch for sample index {idx}: "
                f"expected {self.input_dim}, got {tensor.shape[-1]}"
            )
        return tensor
