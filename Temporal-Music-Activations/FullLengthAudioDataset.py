import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class FullLengthAudioDataset(Dataset):
    """Dataset that loads precomputed feature tensors from .pt or .npy files.

    Expected layout:
    - data_dir can contain files directly or in subdirectories.
    - each sample file should contain either a 1D tensor [D] or 2D tensor [T, D].
    - 2D tensors are reduced to a single vector by averaging over time.
    """

    def __init__(self, data_dir: str, max_samples: int = 0):
        self.data_dir = data_dir
        self.files = self._collect_files(data_dir)
        if max_samples and max_samples > 0:
            self.files = self.files[:max_samples]
        if not self.files:
            raise ValueError(
                f"No .pt or .npy files found under: {data_dir}. "
                "Place precomputed feature tensors there before training."
            )

        sample = self._load_tensor(self.files[0])
        self.input_dim = int(sample.shape[-1])

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
            tensor = torch.load(path, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor in {path}, got {type(tensor)}")
        else:
            array = np.load(path)
            tensor = torch.from_numpy(array)

        tensor = tensor.float()
        if tensor.ndim == 1:
            return tensor
        if tensor.ndim == 2:
            return tensor.mean(dim=0)

        # Collapse unexpected higher-rank tensors to the last feature axis.
        return tensor.reshape(-1, tensor.shape[-1]).mean(dim=0)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        tensor = self._load_tensor(self.files[idx])
        if tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature size mismatch in {self.files[idx]}: "
                f"expected {self.input_dim}, got {tensor.shape[-1]}"
            )
        return tensor
