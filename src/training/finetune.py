"""
Fine-tuning module for SwinIR on satellite imagery.

Implements light fine-tuning with L1 loss only, max 3 epochs,
following the conservative approach to avoid hallucination.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    MODEL_DIR,
    METRICS_DIR,
    SR_SCALE,
)
from src.models.swinir import SwinIRWrapper, load_swinir_model
from src.data.worldstrat_loader import WorldStratDataset, get_validation_loader
from src.metrics.psnr import compute_psnr
from src.metrics.ssim import compute_ssim


class SwinIRTrainer:
    """
    Light fine-tuning trainer for SwinIR on satellite imagery.
    
    Constraints (per project spec):
    - Max 3 epochs
    - L1 loss only (no adversarial/perceptual loss)
    - Abort if training becomes unstable
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-5,
        max_epochs: int = 3,
        patience: int = 2,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: SwinIR model to fine-tune
            device: Training device
            learning_rate: Learning rate (keep low for fine-tuning)
            max_epochs: Maximum training epochs (default: 3)
            patience: Early stopping patience
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.max_epochs = min(max_epochs, 3)  # Hard limit at 3 epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir or MODEL_DIR / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # L1 loss only (no GAN/perceptual loss to avoid hallucination)
        self.criterion = nn.L1Loss()
        
        # Optimizer with low learning rate
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            verbose=True,
        )
        
        # Training state
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = []
        self.is_stable = True
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}")
        
        for batch in pbar:
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            sr = self.model(lr)
            
            # Compute L1 loss
            loss = self.criterion(sr, hr)
            
            # Check for instability
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Training unstable! NaN/Inf loss detected.")
                self.is_stable = False
                return {"loss": float('inf'), "stable": False}
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            "loss": avg_loss,
            "stable": True,
        }
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        num_samples = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)
            
            # Forward pass
            sr = self.model(lr)
            
            # Compute loss
            loss = self.criterion(sr, hr)
            total_loss += loss.item()
            
            # Compute metrics
            sr_np = sr.cpu().numpy()
            hr_np = hr.cpu().numpy()
            
            for i in range(len(sr_np)):
                total_psnr += compute_psnr(sr_np[i], hr_np[i])
                total_ssim += compute_ssim(sr_np[i], hr_np[i])
            
            num_samples += len(sr_np)
        
        return {
            "val_loss": total_loss / max(len(val_loader), 1),
            "val_psnr": total_psnr / max(num_samples, 1),
            "val_ssim": total_ssim / max(num_samples, 1),
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / "swinir_finetuned_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "swinir_finetuned_best.pth"
            torch.save(checkpoint, best_path)
            return best_path
        
        return latest_path
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run complete fine-tuning.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            callback: Optional callback after each epoch
        
        Returns:
            Training results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting SwinIR Fine-tuning")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Loss: L1 (no adversarial)")
        print(f"{'='*60}\n")
        
        for epoch in range(self.max_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Check stability
            if not train_metrics.get("stable", True):
                print("Aborting training due to instability!")
                break
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            self.training_history.append(metrics)
            
            # Print summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {metrics['loss']:.4f}")
            print(f"  Val Loss:   {metrics['val_loss']:.4f}")
            print(f"  Val PSNR:   {metrics['val_psnr']:.2f} dB")
            print(f"  Val SSIM:   {metrics['val_ssim']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(metrics['val_loss'])
            
            # Check for improvement
            is_best = metrics['val_loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['val_loss']
                self.epochs_without_improvement = 0
                print(f"  ✓ New best model!")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping: no improvement for {self.patience} epochs")
                break
            
            # Callback
            if callback:
                callback(epoch, metrics)
        
        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Fine-tuning Complete!")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*60}\n")
        
        return {
            "history": self.training_history,
            "best_loss": self.best_loss,
            "epochs_trained": len(self.training_history),
            "stable": self.is_stable,
        }


def finetune_swinir(
    train_dataset: Optional[WorldStratDataset] = None,
    val_dataset: Optional[WorldStratDataset] = None,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    max_epochs: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Dict[str, Any]:
    """
    Convenience function to fine-tune SwinIR.
    
    Args:
        train_dataset: Training dataset (creates default if None)
        val_dataset: Validation dataset
        batch_size: Training batch size
        learning_rate: Learning rate
        max_epochs: Maximum epochs
        device: Training device
    
    Returns:
        Training results
    """
    # Load datasets
    if train_dataset is None:
        print("Loading WorldStrat training data...")
        train_dataset = WorldStratDataset(split="train", max_samples=500)
    
    if val_dataset is None:
        print("Loading WorldStrat validation data...")
        val_dataset = WorldStratDataset(split="validation", max_samples=100)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Load pretrained model
    print("Loading pretrained SwinIR model...")
    model = load_swinir_model(device=device)
    
    # Create trainer
    trainer = SwinIRTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
    )
    
    # Run training
    results = trainer.train(train_loader, val_loader)
    
    return results


if __name__ == "__main__":
    # Run fine-tuning
    results = finetune_swinir(max_epochs=3)
    print(f"\nTraining complete! Epochs trained: {results['epochs_trained']}")
