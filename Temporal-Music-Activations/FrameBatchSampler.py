import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

from torch.utils.data import BatchSampler, Subset


class GroupedFrameBatchSampler(BatchSampler):
    """Batch sampler that groups frame samples by source file.

    This preserves stride-1 frame coverage while reducing random file access.
    Batches are constructed from contiguous ranges of frame indices belonging to
    the same file, and file order is shuffled each epoch when requested.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle_files: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle_files = shuffle_files
        self.drop_last = drop_last
        self.seed = int(seed)
        self._epoch = 0
        self._file_groups = self._build_file_groups(dataset)
        self._total_samples = sum(len(indices) for indices in self._file_groups.values())
        self._batch_count = self._count_batches()

    def _build_file_groups(self, dataset) -> Dict[int, List[int]]:
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            if not hasattr(base_dataset, "frame_index"):
                raise TypeError("GroupedFrameBatchSampler requires a dataset with frame_index metadata.")

            grouped_pairs: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for subset_position, base_index in enumerate(dataset.indices):
                file_idx, _ = base_dataset.frame_index[base_index]
                grouped_pairs[file_idx].append((int(base_index), subset_position))

            file_groups: Dict[int, List[int]] = {}
            for file_idx, pairs in grouped_pairs.items():
                pairs.sort(key=lambda pair: pair[0])
                file_groups[file_idx] = [subset_position for _, subset_position in pairs]
            return file_groups

        if not hasattr(dataset, "frame_index"):
            raise TypeError("GroupedFrameBatchSampler requires a dataset with frame_index metadata.")

        file_groups: Dict[int, List[int]] = defaultdict(list)
        for dataset_index, (file_idx, _) in enumerate(dataset.frame_index):
            file_groups[int(file_idx)].append(int(dataset_index))
        return file_groups

    def _count_batches(self) -> int:
        if self.drop_last:
            return self._total_samples // self.batch_size
        return math.ceil(self._total_samples / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        file_ids = list(self._file_groups.keys())
        if self.shuffle_files:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(file_ids)
        self._epoch += 1

        tail_pool: List[int] = []

        for file_idx in file_ids:
            file_indices = self._file_groups[file_idx]
            full_batch_stop = len(file_indices) - (len(file_indices) % self.batch_size)

            for start in range(0, full_batch_stop, self.batch_size):
                batch = file_indices[start:start + self.batch_size]
                yield batch

            if full_batch_stop < len(file_indices):
                tail_pool.extend(file_indices[full_batch_stop:])

        if self.drop_last:
            tail_pool = tail_pool[: len(tail_pool) - (len(tail_pool) % self.batch_size)]

        for start in range(0, len(tail_pool), self.batch_size):
            batch = tail_pool[start:start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch

    def __len__(self) -> int:
        return self._batch_count