"""
Model evaluation utilities.

Provides functions to evaluate SR models on validation datasets
and generate comprehensive metrics reports.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import METRICS_DIR, SR_SCALE
from src.metrics.psnr import compute_psnr, psnr_batch
from src.metrics.ssim import compute_ssim, ssim_batch
from src.data.worldstrat_loader import WorldStratDataset, get_validation_loader
from src.models.swinir import SwinIRWrapper
from src.models.bicubic import BicubicUpscaler


def evaluate_model(
    model,
    dataloader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a super-resolution model on a dataset.
    
    Args:
        model: SR model with inference() method
        dataloader: DataLoader providing (lr, hr) pairs
        device: Compute device
        max_samples: Maximum samples to evaluate
    
    Returns:
        Dictionary with evaluation results
    """
    psnr_values = []
    ssim_values = []
    
    model.eval()
    
    n_samples = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        lr = batch['lr']
        hr = batch['hr']
        
        # Move to device
        if isinstance(lr, torch.Tensor):
            lr = lr.to(device)
        
        # Run inference
        with torch.no_grad():
            sr = model.inference(lr)
        
        # Move to CPU for metric computation
        if isinstance(sr, torch.Tensor):
            sr = sr.cpu().numpy()
        if isinstance(hr, torch.Tensor):
            hr = hr.numpy()
        
        # Compute metrics for each sample in batch
        for i in range(len(sr)):
            sr_i = sr[i] if sr.ndim == 4 else sr
            hr_i = hr[i] if hr.ndim == 4 else hr
            
            psnr = compute_psnr(sr_i, hr_i)
            ssim = compute_ssim(sr_i, hr_i)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        
        n_samples += len(sr) if sr.ndim == 4 else 1
        
        if max_samples and n_samples >= max_samples:
            break
    
    # Compute statistics
    psnr_array = np.array(psnr_values)
    ssim_array = np.array(ssim_values)
    
    return {
        "psnr": {
            "mean": float(np.mean(psnr_array)),
            "std": float(np.std(psnr_array)),
            "min": float(np.min(psnr_array)),
            "max": float(np.max(psnr_array)),
        },
        "ssim": {
            "mean": float(np.mean(ssim_array)),
            "std": float(np.std(ssim_array)),
            "min": float(np.min(ssim_array)),
            "max": float(np.max(ssim_array)),
        },
        "n_samples": n_samples,
    }


def evaluate_all_models(
    max_samples: Optional[int] = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate both SwinIR and bicubic baseline on WorldStrat.
    
    Args:
        max_samples: Maximum samples to evaluate
        device: Compute device
    
    Returns:
        Dictionary with results for each model
    """
    print("Loading validation dataset...")
    dataloader = get_validation_loader(batch_size=1, max_samples=max_samples)
    
    results = {}
    
    # Evaluate bicubic baseline
    print("\nEvaluating Bicubic baseline...")
    bicubic = BicubicUpscaler(scale=SR_SCALE)
    results["bicubic"] = evaluate_model(bicubic, dataloader, device, max_samples)
    
    # Evaluate SwinIR
    print("\nEvaluating SwinIR...")
    swinir = SwinIRWrapper(scale=SR_SCALE, device=device)
    
    # Reload dataloader for second evaluation
    dataloader = get_validation_loader(batch_size=1, max_samples=max_samples)
    results["swinir"] = evaluate_model(swinir, dataloader, device, max_samples)
    
    return results


def generate_metrics_report(
    results: Optional[Dict[str, Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
    include_samples: bool = False,
) -> str:
    """
    Generate a formatted metrics report.
    
    Args:
        results: Evaluation results (runs evaluation if None)
        output_path: Path to save report
        include_samples: Whether to include per-sample metrics
    
    Returns:
        Formatted report string
    """
    if results is None:
        results = evaluate_all_models()
    
    # Build report
    lines = []
    lines.append("=" * 60)
    lines.append("SENTINEL-2 SUPER-RESOLUTION METRICS REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Scale: {SR_SCALE}×")
    lines.append("")
    
    # Summary table
    lines.append("-" * 60)
    lines.append("SUMMARY")
    lines.append("-" * 60)
    lines.append("")
    lines.append(f"{'Method':<20} {'PSNR (dB)':<15} {'SSIM':<15}")
    lines.append("-" * 50)
    
    for model_name, model_results in results.items():
        psnr_mean = model_results["psnr"]["mean"]
        ssim_mean = model_results["ssim"]["mean"]
        lines.append(f"{model_name:<20} {psnr_mean:<15.2f} {ssim_mean:<15.4f}")
    
    lines.append("")
    
    # Improvement
    if "bicubic" in results and "swinir" in results:
        psnr_improvement = results["swinir"]["psnr"]["mean"] - results["bicubic"]["psnr"]["mean"]
        ssim_improvement = results["swinir"]["ssim"]["mean"] - results["bicubic"]["ssim"]["mean"]
        
        lines.append("-" * 60)
        lines.append("IMPROVEMENT OVER BASELINE")
        lines.append("-" * 60)
        lines.append(f"PSNR improvement: +{psnr_improvement:.2f} dB")
        lines.append(f"SSIM improvement: +{ssim_improvement:.4f}")
        lines.append("")
    
    # Detailed stats
    lines.append("-" * 60)
    lines.append("DETAILED STATISTICS")
    lines.append("-" * 60)
    
    for model_name, model_results in results.items():
        lines.append(f"\n{model_name.upper()}")
        lines.append(f"  Samples evaluated: {model_results['n_samples']}")
        lines.append(f"  PSNR:")
        lines.append(f"    Mean: {model_results['psnr']['mean']:.2f} dB")
        lines.append(f"    Std:  {model_results['psnr']['std']:.2f} dB")
        lines.append(f"    Min:  {model_results['psnr']['min']:.2f} dB")
        lines.append(f"    Max:  {model_results['psnr']['max']:.2f} dB")
        lines.append(f"  SSIM:")
        lines.append(f"    Mean: {model_results['ssim']['mean']:.4f}")
        lines.append(f"    Std:  {model_results['ssim']['std']:.4f}")
        lines.append(f"    Min:  {model_results['ssim']['min']:.4f}")
        lines.append(f"    Max:  {model_results['ssim']['max']:.4f}")
    
    lines.append("")
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        # Also save as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Report saved to {output_path}")
        print(f"JSON data saved to {json_path}")
    
    return report


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """Print a simple comparison table."""
    print("\n" + "=" * 50)
    print("METRICS COMPARISON")
    print("=" * 50)
    print(f"{'Method':<15} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-" * 50)
    
    for name, res in sorted(results.items()):
        print(f"{name:<15} {res['psnr']['mean']:>8.2f}     {res['ssim']['mean']:>.4f}")
    
    print("=" * 50)


if __name__ == "__main__":
    # Run evaluation and generate report
    print("Running model evaluation on WorldStrat validation set...")
    results = evaluate_all_models(max_samples=50)
    
    # Generate report
    report = generate_metrics_report(
        results,
        output_path=METRICS_DIR / "metrics_report.txt"
    )
    
    print("\n" + report)
