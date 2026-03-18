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
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from FullLengthAudioDataset import FullLengthAudioDataset
from SparseAutoencoder import SparseAutoencoder

class TrainNewSAE:
    def __init__(self, data_dir, output_dir, epochs=100, batch_size=32, learning_rate=1e-3,
                 latent_dim=128, sparsity_weight=1e-5, sparsity_target=0.05, log_interval=10,
                 device='cuda', seed=42, resume=None, checkpoint_interval=10, max_files=0,
                 sample_mode='frames', frame_stride=1, max_frames=0, metrics_filename='training_metrics.csv', verbose=False):
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
        self.sample_mode = sample_mode
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.metrics_filename = metrics_filename
        self.metrics_path = os.path.join(self.output_dir, self.metrics_filename)
        self.verbose = verbose

    def _init_metrics_log(self, append=False):
        """Create metrics CSV with header (or append if resuming and file exists)."""
        os.makedirs(self.output_dir, exist_ok=True)
        file_exists = os.path.exists(self.metrics_path)
        mode = 'a' if append and file_exists else 'w'
        with open(self.metrics_path, mode, newline='') as f:
            writer = csv.writer(f)
            if mode == 'w':
                writer.writerow([
                    'timestamp',
                    'epoch',
                    'avg_loss',
                    'avg_recon',
                    'avg_sparsity',
                    'num_batches',
                ])

    def _append_metrics_log(self, epoch, avg_loss, avg_recon, avg_sparsity, num_batches):
        """Append one row of epoch-level metrics to CSV."""
        with open(self.metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec='seconds'),
                epoch,
                f"{avg_loss:.6f}",
                f"{avg_recon:.6f}",
                f"{avg_sparsity:.6f}",
                num_batches,
            ])

    def _save_checkpoint(self, model, optimizer, epoch, filename):
        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.output_dir, filename)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            checkpoint_path,
        )
        if self.verbose:
            print(f"Saved checkpoint: {checkpoint_path}")

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
        import random
        import numpy as np
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but unavailable; falling back to CPU.")
            self.device = 'cpu'
        if self.device == 'mps':
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if not mps_available:
                print("MPS requested but unavailable; falling back to CPU.")
                self.device = 'cpu'

        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset and create DataLoader
        dataset = FullLengthAudioDataset(
            self.data_dir,
            max_files=self.max_files,
            sample_mode=self.sample_mode,
            frame_stride=self.frame_stride,
            max_frames=self.max_frames,
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print(
            f"Loaded {len(dataset)} training samples from {self.data_dir} "
            f"(sample_mode={self.sample_mode}, input_dim={dataset.input_dim})"
        )
        
        # Initialize SAE model and optimizer
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
            print(f"Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 0

        # Initialize epoch-level metrics CSV in output_dir.
        self._init_metrics_log(append=(start_epoch > 0))
        if self.verbose:
            print(f"Writing training metrics to: {self.metrics_path}")
        
        # Training loop
        for epoch in range(start_epoch, self.epochs):
            model.train()
            total_loss = 0.0
            total_recon = 0.0
            total_sparsity = 0.0
            
            for batch_idx, data in enumerate(dataloader):
                if isinstance(data, (tuple, list)):
                    data = data[0]
                data = data.to(self.device)
                optimizer.zero_grad()
                
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

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_sparsity += sparsity_penalty.item()

                if self.verbose and (batch_idx + 1) % self.log_interval == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}] "
                        f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                        f"Loss: {loss.item():.6f} "
                        f"Recon: {recon_loss.item():.6f} "
                        f"Sparsity: {sparsity_penalty.item():.6f}"
                    )

            num_batches = max(len(dataloader), 1)
            avg_loss = total_loss / num_batches
            avg_recon = total_recon / num_batches
            avg_sparsity = total_sparsity / num_batches
            print(
                f"Epoch [{epoch + 1}/{self.epochs}] "
                f"Avg Loss: {avg_loss:.6f} "
                f"Avg Recon: {avg_recon:.6f} "
                f"Avg Sparsity: {avg_sparsity:.6f}"
            )

            self._append_metrics_log(
                epoch=epoch + 1,
                avg_loss=avg_loss,
                avg_recon=avg_recon,
                avg_sparsity=avg_sparsity,
                num_batches=num_batches,
            )

            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(model, optimizer, epoch, f"checkpoint_epoch_{epoch + 1}.pt")

        self._save_checkpoint(model, optimizer, self.epochs - 1, "sae_final.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on full-length audio tracks.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training feature tensors.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Optimizer learning rate.')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension size for the SAE.')
    parser.add_argument('--sparsity_weight', type=float, default=1e-5, help='Weight of KL sparsity regularizer.')
    parser.add_argument('--sparsity_target', type=float, default=0.05, help='Target activation sparsity level.')
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N batches when verbose.')
    parser.add_argument('--device', type=str, default='cuda', help="Training device, e.g. 'cpu' or 'cuda'.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs.')
    parser.add_argument('--max_files', type=int, default=0, help='Limit the number of feature files loaded for quick runs (0 = no limit).')
    parser.add_argument('--sample_mode', type=str, default='frames', choices=['frames', 'mean'], help='Use every timestep vector or mean-pool each track before training.')
    parser.add_argument('--frame_stride', type=int, default=1, help='Use every Nth timestep when sample_mode=frames.')
    parser.add_argument('--max_frames', type=int, default=0, help='Optional cap on total timestep samples when sample_mode=frames.')
    parser.add_argument('--metrics_filename', type=str, default='training_metrics.csv', help='CSV filename for epoch-level training metrics in output_dir.')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed per-batch logging.')
    return parser.parse_args()


def main():
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
        sample_mode=args.sample_mode,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        metrics_filename=args.metrics_filename,
        verbose=args.verbose,
    )
    trainer.train()


if __name__ == '__main__':
    main()