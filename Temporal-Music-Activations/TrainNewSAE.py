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
import os
import torch
from torch.utils.data import DataLoader
from FullLengthAudioDataset import FullLengthAudioDataset
from SparseAutoencoder import SparseAutoencoder

class TrainNewSAE:
    def __init__(self, data_dir, output_dir, epochs=100, batch_size=32, learning_rate=1e-3,
                 latent_dim=128, sparsity_weight=1e-5, sparsity_target=0.05, log_interval=10,
                 device='cuda', seed=42, resume=None, checkpoint_interval=10, max_samples=0,
                 sample_mode='frames', frame_stride=1, max_frames=0, verbose=False):
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
        self.max_samples = max_samples
        self.sample_mode = sample_mode
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.verbose = verbose

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

    def _compute_sparsity_penalty(self, latent_activations):
        # Estimate feature firing probabilities in [0, 1] for KL sparsity.
        rho_hat = torch.sigmoid(latent_activations).mean(dim=0)
        rho_hat = torch.clamp(rho_hat, min=1e-6, max=1 - 1e-6)
        rho = torch.full_like(rho_hat, fill_value=self.sparsity_target)
        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        return torch.sum(kl)
        
    def train(self):
        # Set random seed for reproducibility
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
            max_samples=self.max_samples,
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
            checkpoint = torch.load(self.resume, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 0
        
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
            print(
                f"Epoch [{epoch + 1}/{self.epochs}] "
                f"Avg Loss: {total_loss / num_batches:.6f} "
                f"Avg Recon: {total_recon / num_batches:.6f} "
                f"Avg Sparsity: {total_sparsity / num_batches:.6f}"
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
    parser.add_argument('--max_samples', type=int, default=0, help='Limit the number of feature files for quick runs.')
    parser.add_argument('--sample_mode', type=str, default='frames', choices=['frames', 'mean'], help='Use every timestep vector or mean-pool each track before training.')
    parser.add_argument('--frame_stride', type=int, default=1, help='Use every Nth timestep when sample_mode=frames.')
    parser.add_argument('--max_frames', type=int, default=0, help='Optional cap on total timestep samples when sample_mode=frames.')
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
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        verbose=args.verbose,
    )
    trainer.train()


if __name__ == '__main__':
    main()