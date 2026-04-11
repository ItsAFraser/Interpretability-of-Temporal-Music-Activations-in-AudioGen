'''
This script is intended to train a new Sparse Autoencoder (SAE) on longer audio files from full-length tracks. The goal is to allow the SAE to learn long-range musical features that may not be captured when training on shorter clips, as done in previous works.

Usage:
    python TrainNewSAE.py --data_dir /path/to/full_length_tracks --output_dir /path/to/save_model
Arguments:
    --data_dir: Directory containing precomputed MusicGen feature tensors for training.
    --output_dir: Directory where the trained SAE model will be saved.  
    --epochs: Number of training epochs.
    --batch_size: Batch size for training.
    --learning_rate: Learning rate for the optimizer.
    --latent_dim: Dimensionality of the latent space in the SAE.
    --sparsity_weight: Weight for the sparsity penalty in the loss function.
    --sparsity_target: Target sparsity level for the activations.
    --log_interval: Interval (in batches) at which to log training progress.
    --device: Device to use for training (e.g., 'cpu' or 'cuda').
    --seed: Random seed for reproducibility.
    --resume: Path to a checkpoint to resume training from.
    --checkpoint_interval: Interval (in epochs) at which to save model checkpoints.
    --sample_mode: 'frames' keeps timestep vectors; 'mean' averages each track first.
    --frame_stride: Use every Nth timestep when sample_mode='frames'.
    --max_frames: Optional cap on total timestep samples.
    --verbose: If set, print detailed training progress.
'''
import argparse
import csv
import json
import os
import random
from datetime import datetime
from time import perf_counter
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from FrameBatchSampler import GroupedFrameBatchSampler
from FullLengthAudioDataset import FullLengthAudioDataset
from SparseAutoencoder import SparseAutoencoder

class TrainNewSAE:
    """Train a Sparse Autoencoder over MusicGen feature tensors.

    The trainer supports:
    - Frame-level or track-level (mean pooled) sampling.
    - Deterministic train/validation splitting.
    - Epoch-level metric logging to CSV.
    - Periodic checkpoints plus best/final checkpoints.
    - Resume-from-checkpoint training.
    """

    def __init__(self, data_dir, output_dir, epochs=100, batch_size=32, learning_rate=1e-3,
                 latent_dim=128, sparsity_weight=1e-5, sparsity_target=0.05, log_interval=10,
                 device='cuda', seed=42, resume=None, checkpoint_interval=10, max_files=0,
                 random_subset_files=0, file_manifest_path=None, source_data_dir=None, subset_seed=42,
                 sampler_mode='random',
                 sample_mode='frames', frame_stride=1, max_frames=0, metrics_filename='training_metrics.csv',
                 val_split=0.1, num_workers=0, prefetch_factor=4, persistent_workers=None,
                 save_best=True, profile_timing=True, verbose=False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        self.log_interval = log_interval
        self.device = device
        self.seed = seed
        self.resume = resume
        self.checkpoint_interval = checkpoint_interval
        self.max_files = max_files
        self.random_subset_files = random_subset_files
        self.file_manifest_path = file_manifest_path
        self.source_data_dir = source_data_dir or data_dir
        self.subset_seed = subset_seed
        self.sampler_mode = sampler_mode
        self.sample_mode = sample_mode
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.metrics_filename = metrics_filename
        self.val_split = val_split
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.save_best = save_best
        self.profile_timing = profile_timing
        self.metrics_path = os.path.join(self.output_dir, self.metrics_filename)
        self.run_manifest_path = os.path.join(self.output_dir, 'run_manifest.json')
        self.verbose = verbose

        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be positive, got {self.prefetch_factor}")
        if self.sampler_mode not in {'random', 'grouped'}:
            raise ValueError(f"sampler_mode must be 'random' or 'grouped', got {self.sampler_mode!r}")

    def _init_metrics_log(self, append=False):
        """Create the epoch metrics CSV.

        When resuming, we append to an existing file (if present) so one CSV can
        capture the full run history across multiple invocations.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        file_exists = os.path.exists(self.metrics_path)
        mode = 'a' if append and file_exists else 'w'
        with open(self.metrics_path, mode, newline='') as f:
            writer = csv.writer(f)
            if mode == 'w':
                writer.writerow([
                    'timestamp',
                    'epoch',
                    'train_loss',
                    'train_recon',
                    'train_sparsity',
                    'train_batches',
                    'train_fetch_sec',
                    'train_transfer_sec',
                    'train_compute_sec',
                    'train_epoch_sec',
                    'train_samples_per_sec',
                    'val_loss',
                    'val_recon',
                    'val_sparsity',
                    'val_batches',
                ])

    def _append_metrics_log(
        self,
        epoch,
        train_loss,
        train_recon,
        train_sparsity,
        train_batches,
        train_fetch_sec,
        train_transfer_sec,
        train_compute_sec,
        train_epoch_sec,
        train_samples_per_sec,
        val_loss,
        val_recon,
        val_sparsity,
        val_batches,
    ):
        """Append one epoch summary row to the metrics CSV."""
        with open(self.metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec='seconds'),
                epoch,
                f"{train_loss:.6f}",
                f"{train_recon:.6f}",
                f"{train_sparsity:.6f}",
                train_batches,
                f"{train_fetch_sec:.6f}",
                f"{train_transfer_sec:.6f}",
                f"{train_compute_sec:.6f}",
                f"{train_epoch_sec:.6f}",
                f"{train_samples_per_sec:.6f}",
                '' if val_loss is None else f"{val_loss:.6f}",
                '' if val_recon is None else f"{val_recon:.6f}",
                '' if val_sparsity is None else f"{val_sparsity:.6f}",
                val_batches,
            ])

    def _save_checkpoint(self, model, optimizer, epoch, filename, best_metric=None):
        """Persist model and optimizer state.

        Storing optimizer state is important for exact resumption because Adam's
        running moments materially affect future updates.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.output_dir, filename)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
            },
            checkpoint_path,
        )
        if self.verbose:
            print(f"Saved checkpoint: {checkpoint_path}")

    def _write_run_manifest(
        self,
        dataset_size,
        train_size,
        val_size,
        input_dim,
        selected_relative_files,
        data_format,
        repacked_metadata_path,
    ):
        """Write a run manifest for reproducibility and experiment tracking."""
        pin_memory, persistent_workers, prefetch_factor = self._resolve_dataloader_settings()
        manifest = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'source_data_dir': self.source_data_dir,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'latent_dim': self.latent_dim,
            'sparsity_weight': self.sparsity_weight,
            'sparsity_target': self.sparsity_target,
            'device': self.device,
            'seed': self.seed,
            'sample_mode': self.sample_mode,
            'sampler_mode': self.sampler_mode,
            'data_format': data_format,
            'frame_stride': self.frame_stride,
            'max_frames': self.max_frames,
            'max_files': self.max_files,
            'random_subset_files': self.random_subset_files,
            'file_manifest_path': self.file_manifest_path,
            'repacked_metadata_path': repacked_metadata_path,
            'subset_seed': self.subset_seed,
            'val_split': self.val_split,
            'num_workers': self.num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
            'prefetch_factor': prefetch_factor,
            'profile_timing': self.profile_timing,
            'dataset_size': dataset_size,
            'train_size': train_size,
            'val_size': val_size,
            'selected_files': len(selected_relative_files),
            'selected_relative_files_path': self.file_manifest_path,
            'input_dim': input_dim,
            'metrics_path': self.metrics_path,
        }
        with open(self.run_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

    def _resolve_dataloader_settings(self):
        pin_memory = (self.device == 'cuda')
        persistent_workers = self.num_workers > 0
        if self.persistent_workers is not None:
            persistent_workers = bool(self.persistent_workers) and self.num_workers > 0
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        return pin_memory, persistent_workers, prefetch_factor

    def _build_dataloader(self, dataset, shuffle):
        """Build a deterministic DataLoader for reproducible experiments."""
        # Use an explicit generator so DataLoader shuffling is seed-controlled.
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        def _seed_worker(worker_id):
            # Keep RNGs deterministic across worker processes.
            worker_seed = self.seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        pin_memory, persistent_workers, prefetch_factor = self._resolve_dataloader_settings()

        use_grouped_sampler = (
            self.sampler_mode == 'grouped'
            and self.sample_mode == 'frames'
            and hasattr(dataset, '__len__')
            and len(dataset) > 0
        )

        if use_grouped_sampler:
            batch_sampler = GroupedFrameBatchSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle_files=shuffle,
                drop_last=False,
                seed=self.seed,
            )
            dataloader_kwargs = {
                'batch_sampler': batch_sampler,
                'num_workers': self.num_workers,
                'pin_memory': pin_memory,
                'worker_init_fn': _seed_worker if self.num_workers > 0 else None,
            }
            if self.num_workers > 0:
                dataloader_kwargs['persistent_workers'] = persistent_workers
                if prefetch_factor is not None:
                    dataloader_kwargs['prefetch_factor'] = prefetch_factor
            return DataLoader(dataset, **dataloader_kwargs)

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': shuffle,
            'num_workers': self.num_workers,
            'pin_memory': pin_memory,
            'worker_init_fn': _seed_worker if self.num_workers > 0 else None,
            'generator': generator,
        }
        if self.num_workers > 0:
            dataloader_kwargs['persistent_workers'] = persistent_workers
            if prefetch_factor is not None:
                dataloader_kwargs['prefetch_factor'] = prefetch_factor

        return DataLoader(dataset, **dataloader_kwargs)

    def _sync_device_for_timing(self):
        if not self.profile_timing:
            return
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elif self.device == 'mps' and hasattr(torch, 'mps'):
            torch.mps.synchronize()

    @torch.no_grad()
    def _evaluate(self, model, dataloader) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
        """Evaluate one full pass over a dataloader.

        Returns average total loss, reconstruction loss, sparsity penalty, and
        number of batches. Returns Nones when dataloader is disabled.
        """
        if dataloader is None:
            return None, None, None, 0

        model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_sparsity = 0.0

        for data in dataloader:
            if isinstance(data, (tuple, list)):
                data = data[0]
            data = data.to(self.device)

            # Forward pass through the SAE without tracking gradients. loss is computed for monitoring but not backpropagated.
            recon_data, latent_activations = model(data)
            recon_loss = torch.nn.functional.mse_loss(recon_data, data)
            sparsity_penalty = self._compute_sparsity_penalty(latent_activations)
            loss = recon_loss + self.sparsity_weight * sparsity_penalty

            # Accumulate batch losses for epoch-level averaging.
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_sparsity += sparsity_penalty.item()

        num_batches = max(len(dataloader), 1)
        return (
            total_loss / num_batches,
            total_recon / num_batches,
            total_sparsity / num_batches,
            num_batches,
        )

    def _compute_sparsity_penalty(self, latent_activations: torch.Tensor) -> torch.Tensor:
        # L1 penalty on post-ReLU activations is the standard SAE sparsity regulariser.
        # The previous KL formulation applied sigmoid() to post-ReLU values, which maps
        # all activations to [0.5, 1] — making rho_hat always ≥ 0.5 regardless of the
        # sparsity_target, so the KL penalty was incoherent throughout training.
        # L1 (mean activation) is differentiable, scale-invariant, and widely used
        # in mechanistic-interpretability SAE work (e.g. Anthropic, EleutherAI).
        if latent_activations.ndim != 2:
            raise ValueError(
                f"_compute_sparsity_penalty expects a 2-D [batch, latent_dim] tensor, "
                f"got shape {tuple(latent_activations.shape)}"
            )
        return latent_activations.mean()
        
    def train(self):
        # Set all random seeds for full reproducibility across torch, numpy, and
        # Python's random module (the last is used by DataLoader shuffle).
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but unavailable; falling back to CPU.")
            self.device = 'cpu'
        if self.device == 'mps':
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if not mps_available:
                print("MPS requested but unavailable; falling back to CPU.")
                self.device = 'cpu'

        os.makedirs(self.output_dir, exist_ok=True)
        
        # Build the flattened training dataset from feature files.
        # Depending on sample_mode, each sample is either:
        # - one timestep/frame vector (frames mode), or
        # - one mean-pooled track vector (mean mode).
        repacked_metadata_path = os.path.join(self.data_dir, 'repacked_metadata.json')
        if not os.path.isfile(repacked_metadata_path):
            repacked_metadata_path = None

        dataset = FullLengthAudioDataset(
            self.data_dir,
            max_files=self.max_files,
            random_subset_files=self.random_subset_files,
            file_manifest_path=self.file_manifest_path,
            repacked_metadata_path=repacked_metadata_path,
            subset_seed=self.subset_seed,
            sample_mode=self.sample_mode,
            frame_stride=self.frame_stride,
            max_frames=self.max_frames,
        )

        # Deterministically split train/val unless validation is disabled or the
        # dataset is too small to support a meaningful split.
        if len(dataset) < 2 or self.val_split <= 0:
            train_dataset = dataset
            val_dataset = None
        else:
            val_size = max(1, int(len(dataset) * self.val_split))
            train_size = len(dataset) - val_size
            if train_size <= 0:
                train_size = len(dataset) - 1
                val_size = 1

            split_gen = torch.Generator().manual_seed(self.seed)
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=split_gen,
            )

        # Shuffle only training data. Validation order should remain stable.
        pin_memory, persistent_workers, prefetch_factor = self._resolve_dataloader_settings()
        train_dataloader = self._build_dataloader(train_dataset, shuffle=True)
        val_dataloader = self._build_dataloader(val_dataset, shuffle=False) if val_dataset is not None else None

        print(
            f"Loaded {len(dataset)} training samples from {self.data_dir} "
            f"(sample_mode={self.sample_mode}, input_dim={dataset.input_dim})"
        )
        print(
            f"Selected {dataset.selected_files} feature files from {dataset.total_files_discovered} discovered "
            f"(max_files={self.max_files}, random_subset_files={self.random_subset_files}, subset_seed={self.subset_seed})"
        )
        if val_dataset is not None:
            print(f"Train/val split: {len(train_dataset)} / {len(val_dataset)}")
        else:
            print("Validation disabled (dataset too small or val_split <= 0).")
        print(
            "DataLoader settings: "
            f"num_workers={self.num_workers}, "
            f"pin_memory={pin_memory}, "
            f"persistent_workers={persistent_workers}, "
            f"prefetch_factor={prefetch_factor}, "
            f"sampler_mode={self.sampler_mode}, "
            f"profile_timing={self.profile_timing}"
        )

        self._write_run_manifest(
            dataset_size=len(dataset),
            train_size=len(train_dataset),
            val_size=0 if val_dataset is None else len(val_dataset),
            input_dim=dataset.input_dim,
            selected_relative_files=dataset.get_selected_relative_files(),
            data_format=dataset.data_format,
            repacked_metadata_path=dataset.repacked_metadata_path,
        )
        
        # Initialize SAE and optimizer after dataset load so input_dim can be
        # derived directly from data.
        model = SparseAutoencoder(input_dim=dataset.input_dim, latent_dim=self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Optionally resume from a checkpoint
        if self.resume:
            if not os.path.isfile(self.resume):
                raise FileNotFoundError(f"Resume checkpoint not found: {self.resume}")
            checkpoint = torch.load(self.resume, map_location=self.device, weights_only=False)
            if 'model_state_dict' not in checkpoint or 'optimizer_state_dict' not in checkpoint:
                raise KeyError(
                    f"Checkpoint {self.resume} is missing required keys "
                    "('model_state_dict', 'optimizer_state_dict')."
                )
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            # Carry forward best checkpoint metric so future epochs compare
            # against the correct historical value.
            best_metric = checkpoint.get('best_metric')
            print(f"Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 0
            best_metric = None

        # Initialize epoch-level metrics CSV in output_dir.
        self._init_metrics_log(append=(start_epoch > 0))
        if self.verbose:
            print(f"Writing training metrics to: {self.metrics_path}")
        
        # Main optimization loop.
        for epoch in range(start_epoch, self.epochs):
            model.train()
            total_loss = 0.0
            total_recon = 0.0
            total_sparsity = 0.0
            total_fetch_sec = 0.0
            total_transfer_sec = 0.0
            total_compute_sec = 0.0
            total_train_samples = 0
            epoch_start_time = perf_counter()

            train_iterator = iter(train_dataloader)
            batch_idx = 0
            while True:
                fetch_start_time = perf_counter()
                try:
                    data = next(train_iterator)
                except StopIteration:
                    break
                total_fetch_sec += perf_counter() - fetch_start_time

                if isinstance(data, (tuple, list)):
                    data = data[0]
                transfer_start_time = perf_counter()
                data = data.to(self.device, non_blocking=pin_memory)
                self._sync_device_for_timing()
                total_transfer_sec += perf_counter() - transfer_start_time
                total_train_samples += int(data.shape[0])

                optimizer.zero_grad(set_to_none=True)
                compute_start_time = perf_counter()
                
                # Forward pass through the SAE
                recon_data, latent_activations = model(data)
                
                # Compute reconstruction loss and sparsity penalty
                recon_loss = torch.nn.functional.mse_loss(recon_data, data)
                sparsity_penalty = self._compute_sparsity_penalty(latent_activations)
                loss = recon_loss + self.sparsity_weight * sparsity_penalty

                loss.backward()
                optimizer.step()
                # Keep decoder columns at unit norm so the encoder is forced to
                # be sparse rather than pushing magnitude into the decoder.
                model.normalize_decoder()
                self._sync_device_for_timing()
                total_compute_sec += perf_counter() - compute_start_time

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_sparsity += sparsity_penalty.item()

                if self.verbose and (batch_idx + 1) % self.log_interval == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}] "
                        f"Batch [{batch_idx + 1}/{len(train_dataloader)}] "
                        f"Loss: {loss.item():.6f} "
                        f"Recon: {recon_loss.item():.6f} "
                        f"Sparsity: {sparsity_penalty.item():.6f}"
                    )
                batch_idx += 1

            train_batches = max(len(train_dataloader), 1)
            avg_loss = total_loss / train_batches
            avg_recon = total_recon / train_batches
            avg_sparsity = total_sparsity / train_batches
            train_epoch_sec = perf_counter() - epoch_start_time
            train_samples_per_sec = total_train_samples / max(train_epoch_sec, 1e-8)

            # Evaluate at epoch boundary for a stable model-selection signal.
            val_loss, val_recon, val_sparsity, val_batches = self._evaluate(model, val_dataloader)

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] "
                f"Avg Loss: {avg_loss:.6f} "
                f"Avg Recon: {avg_recon:.6f} "
                f"Avg Sparsity: {avg_sparsity:.6f}"
            )
            print(
                f"Epoch [{epoch + 1}/{self.epochs}] "
                f"Timing: fetch={total_fetch_sec:.2f}s "
                f"transfer={total_transfer_sec:.2f}s "
                f"compute={total_compute_sec:.2f}s "
                f"epoch={train_epoch_sec:.2f}s "
                f"samples/s={train_samples_per_sec:.2f}"
            )
            if val_loss is not None:
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}] "
                    f"Val Loss: {val_loss:.6f} "
                    f"Val Recon: {val_recon:.6f} "
                    f"Val Sparsity: {val_sparsity:.6f}"
                )

            self._append_metrics_log(
                epoch=epoch + 1,
                train_loss=avg_loss,
                train_recon=avg_recon,
                train_sparsity=avg_sparsity,
                train_batches=train_batches,
                train_fetch_sec=total_fetch_sec,
                train_transfer_sec=total_transfer_sec,
                train_compute_sec=total_compute_sec,
                train_epoch_sec=train_epoch_sec,
                train_samples_per_sec=train_samples_per_sec,
                val_loss=val_loss,
                val_recon=val_recon,
                val_sparsity=val_sparsity,
                val_batches=val_batches,
            )

            # Prefer validation loss for checkpoint selection when available.
            # If validation is disabled, fall back to train loss.
            monitor_metric = val_loss if val_loss is not None else avg_loss
            if self.save_best and (best_metric is None or monitor_metric < best_metric):
                best_metric = monitor_metric
                self._save_checkpoint(model, optimizer, epoch, 'sae_best.pt', best_metric=best_metric)
                if self.verbose:
                    print(
                        f"New best checkpoint at epoch {epoch + 1}: "
                        f"metric={best_metric:.6f}"
                    )

            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    f"checkpoint_epoch_{epoch + 1}.pt",
                    best_metric=best_metric,
                )

        # Always save a final checkpoint for completeness, even when not the best.
        self._save_checkpoint(model, optimizer, self.epochs - 1, 'sae_final.pt', best_metric=best_metric)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on full-length audio tracks.")
    # Data and output locations.
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training feature tensors.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints.')

    # Core optimization hyperparameters.
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Optimizer learning rate.')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension size for the SAE.')
    parser.add_argument('--sparsity_weight', type=float, default=1e-5, help='Weight of L1 sparsity regularizer.')
    parser.add_argument('--sparsity_target', type=float, default=0.05, help='Target activation sparsity level.')

    # Logging/checkpoint behavior.
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N batches when verbose.')
    parser.add_argument('--device', type=str, default='cuda', help="Training device, e.g. 'cpu' or 'cuda'.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs.')

    # Dataset sampling controls.
    parser.add_argument('--max_files', type=int, default=0, help='Limit the number of feature files loaded for quick runs (0 = no limit).')
    parser.add_argument('--random_subset_files', type=int, default=0, help='Randomly choose this many feature files before frame expansion (0 = use all files).')
    parser.add_argument('--file_manifest_path', type=str, default=None, help='Optional manifest containing one selected relative feature path per line.')
    parser.add_argument('--source_data_dir', type=str, default=None, help='Original source layer directory used to resolve a staged file manifest.')
    parser.add_argument('--subset_seed', type=int, default=42, help='Seed used for deterministic random file subset selection.')
    parser.add_argument('--sampler_mode', type=str, default='random', choices=['random', 'grouped'], help='Batch construction mode for frame training. grouped keeps batches within a small number of files to reduce random I/O.')
    parser.add_argument('--sample_mode', type=str, default='frames', choices=['frames', 'mean'], help='Use every timestep vector or mean-pool each track before training.')
    parser.add_argument('--frame_stride', type=int, default=1, help='Use every Nth timestep when sample_mode=frames.')
    parser.add_argument('--max_frames', type=int, default=0, help='Optional cap on total timestep samples when sample_mode=frames.')

    # Validation/performance controls.
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of samples used for validation. Set 0 to disable.')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker processes. Increase on HPC for faster I/O.')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='How many batches each worker preloads ahead when num_workers > 0.')
    parser.add_argument('--persistent_workers', action='store_true', dest='persistent_workers', help='Keep DataLoader workers alive across epochs when num_workers > 0.')
    parser.add_argument('--no_persistent_workers', action='store_false', dest='persistent_workers', help='Disable persistent DataLoader workers.')
    parser.add_argument('--save_best', action='store_true', help='Save best checkpoint as sae_best.pt using val_loss (or train_loss without validation).')
    parser.add_argument('--no_save_best', action='store_false', dest='save_best', help='Disable saving sae_best.pt.')
    parser.add_argument('--profile_timing', action='store_true', dest='profile_timing', help='Record and print epoch timing breakdowns for data fetch, transfer, and compute.')
    parser.add_argument('--no_profile_timing', action='store_false', dest='profile_timing', help='Disable timing breakdown instrumentation.')
    parser.add_argument('--metrics_filename', type=str, default='training_metrics.csv', help='CSV filename for epoch-level training metrics in output_dir.')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed per-batch logging.')
    parser.set_defaults(save_best=True, persistent_workers=None, profile_timing=True)
    return parser.parse_args()


def main():
    # Parse CLI, construct trainer, and run end-to-end training.
    args = parse_args()
    trainer = TrainNewSAE(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        sparsity_weight=args.sparsity_weight,
        sparsity_target=args.sparsity_target,
        log_interval=args.log_interval,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        max_files=args.max_files,
        random_subset_files=args.random_subset_files,
        file_manifest_path=args.file_manifest_path,
        source_data_dir=args.source_data_dir,
        subset_seed=args.subset_seed,
        sampler_mode=args.sampler_mode,
        sample_mode=args.sample_mode,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        val_split=args.val_split,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        save_best=args.save_best,
        profile_timing=args.profile_timing,
        metrics_filename=args.metrics_filename,
        verbose=args.verbose,
    )
    trainer.train()


if __name__ == '__main__':
    main()