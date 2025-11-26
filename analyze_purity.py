"""
Feature Purity Analysis Script

This script analyzes neuron cosine similarity across different checkpoints
to quantify feature purity changes due to misalignment.

Usage:
    python analyze_purity.py --checkpoints Cs0:path1 Cs10:path2 --output results.csv
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from models import CLIP_VITB16


def load_model(ckpt_path):
    """Load CLIP model from checkpoint"""
    print(f"Loading model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    
    model = CLIP_VITB16(rand_embed=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model


def normalize_tensor(V):
    """Normalize vectors to unit length"""
    norm = torch.norm(V, p=2, dim=1, keepdim=True)
    return V / norm


def compute_purity_metrics(projection):
    """
    Compute multiple purity metrics for projection layer
    
    Args:
        projection: Weight matrix of shape (input_dim, output_dim), e.g., (768, 512)
    
    Returns:
        dict: Dictionary containing various purity metrics
    """
    # Transpose to get neurons as rows: (512, 768)
    V = projection.T
    V_norm = normalize_tensor(V)
    
    # Compute cosine similarity matrix: (512, 512)
    cosine_sim = torch.matmul(V_norm, V_norm.T)
    
    # Remove diagonal (self-similarity)
    cosine_sim_no_diag = cosine_sim - torch.diag(torch.diag(cosine_sim))
    
    # Compute metrics
    metrics = {
        'avg_abs_cos_sim': torch.mean(torch.abs(cosine_sim_no_diag)).item(),
        'max_abs_cos_sim': torch.max(torch.abs(cosine_sim_no_diag)).item(),
        'avg_cos_sim': torch.mean(cosine_sim_no_diag).item(),
        'std_cos_sim': torch.std(cosine_sim_no_diag).item(),
        'median_abs_cos_sim': torch.median(torch.abs(cosine_sim_no_diag)).item(),
    }
    
    # Compute per-neuron purity (inverse of avg abs similarity)
    per_neuron_sim = torch.mean(torch.abs(cosine_sim_no_diag), dim=1)
    metrics['per_neuron_purity_mean'] = (1.0 / per_neuron_sim.mean()).item()
    metrics['per_neuron_purity_std'] = torch.std(1.0 / per_neuron_sim).item()
    
    return metrics, cosine_sim_no_diag.numpy()


def plot_cosine_histogram(cosine_matrices, labels, output_path):
    """Plot histogram of cosine similarities for multiple models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (cos_matrix, label) in enumerate(zip(cosine_matrices, labels)):
        if idx >= len(axes):
            break
        
        # Flatten upper triangle (avoid double counting)
        mask = np.triu(np.ones_like(cos_matrix, dtype=bool), k=1)
        cos_values = cos_matrix[mask]
        
        axes[idx].hist(cos_values, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Orthogonal')
        axes[idx].set_xlabel('Cosine Similarity', fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'{label}\nMean: {cos_values.mean():.3f}', fontsize=12)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Remove unused subplots
    for idx in range(len(cosine_matrices), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved histogram to {output_path}")
    plt.close()


def plot_purity_vs_misalignment(df, output_path):
    """Plot purity metrics vs misalignment level"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Average absolute cosine similarity
    axes[0].plot(df['Cs'], df['avg_abs_cos_sim'], 'b-o', linewidth=3, markersize=10)
    axes[0].set_xlabel('Shuffling Probability (Cs)', fontsize=14)
    axes[0].set_ylabel('Avg Absolute Cosine Similarity', fontsize=14)
    axes[0].set_title('Feature Purity vs Misalignment', fontsize=16, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=12)
    
    # Plot 2: Max absolute cosine similarity
    axes[1].plot(df['Cs'], df['max_abs_cos_sim'], 'r-s', linewidth=3, markersize=10)
    axes[1].set_xlabel('Shuffling Probability (Cs)', fontsize=14)
    axes[1].set_ylabel('Max Absolute Cosine Similarity', fontsize=14)
    axes[1].set_title('Worst-Case Similarity vs Misalignment', fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved purity plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze feature purity across checkpoints')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                       help='Checkpoints in format "label:path", e.g., "Cs0:checkpoint.pt Cs10:path/to/ckpt.pt"')
    parser.add_argument('--output-csv', default='purity_analysis.csv',
                       help='Output CSV file for numerical results')
    parser.add_argument('--output-plot', default='purity_vs_cs.png',
                       help='Output plot file')
    parser.add_argument('--output-hist', default='cosine_histograms.png',
                       help='Output histogram file')
    parser.add_argument('--projection', choices=['text', 'image', 'both'], default='text',
                       help='Which projection to analyze')
    
    args = parser.parse_args()
    
    # Parse checkpoints
    checkpoints = {}
    for item in args.checkpoints:
        if ':' not in item:
            print(f"Error: Checkpoint must be in format 'label:path', got '{item}'")
            sys.exit(1)
        label, path = item.split(':', 1)
        checkpoints[label] = path
    
    print(f"\nAnalyzing {len(checkpoints)} checkpoints...")
    print(f"Projection layer: {args.projection}")
    print("-" * 60)
    
    results = []
    cosine_matrices = []
    labels = []
    
    for label, ckpt_path in sorted(checkpoints.items()):
        print(f"\n{label}:")
        model = load_model(ckpt_path)
        
        # Extract Cs value from label (e.g., "Cs30" -> 0.3)
        try:
            if label.lower().startswith('cs'):
                cs_value = float(label[2:]) / 100
            else:
                cs_value = 0.0
        except:
            cs_value = 0.0
        
        result_dict = {'label': label, 'Cs': cs_value, 'checkpoint': ckpt_path}
        
        if args.projection in ['text', 'both']:
            print("  Analyzing text projection...")
            text_metrics, text_cos = compute_purity_metrics(model.text_projection)
            for key, val in text_metrics.items():
                result_dict[f'text_{key}'] = val
            
            if args.projection == 'text':
                cosine_matrices.append(text_cos)
                labels.append(f"{label} (Text)")
        
        if args.projection in ['image', 'both']:
            print("  Analyzing image projection...")
            image_metrics, image_cos = compute_purity_metrics(model.image_projection)
            for key, val in image_metrics.items():
                result_dict[f'image_{key}'] = val
            
            if args.projection == 'image':
                cosine_matrices.append(image_cos)
                labels.append(f"{label} (Image)")
        
        if args.projection == 'both':
            # Average of text and image for histograms
            avg_cos = (text_cos + image_cos) / 2
            cosine_matrices.append(avg_cos)
            labels.append(f"{label} (Avg)")
        
        results.append(result_dict)
        
        # Print key metrics
        print(f"  Text Avg Abs Cos Sim: {result_dict.get('text_avg_abs_cos_sim', 'N/A'):.4f}")
        if 'image_avg_abs_cos_sim' in result_dict:
            print(f"  Image Avg Abs Cos Sim: {result_dict['image_avg_abs_cos_sim']:.4f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Cs')
    
    # Save results
    df.to_csv(args.output_csv, index=False, float_format='%.6f')
    print(f"\n✓ Results saved to {args.output_csv}")
    
    # Generate plots
    if len(df) > 1:
        plot_purity_vs_misalignment(df, args.output_plot)
        plot_cosine_histogram(cosine_matrices, labels, args.output_hist)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    proj_name = 'text_avg_abs_cos_sim' if args.projection == 'text' else 'image_avg_abs_cos_sim'
    if args.projection == 'both':
        proj_name = 'text_avg_abs_cos_sim'
    
    if proj_name in df.columns and len(df) > 1:
        baseline = df[df['Cs'] == 0][proj_name].values[0] if any(df['Cs'] == 0) else df[proj_name].iloc[0]
        print(f"\nBaseline purity (Cs=0): {baseline:.4f}")
        print("\nRelative changes:")
        for _, row in df.iterrows():
            if row['Cs'] > 0:
                purity = row[proj_name]
                change = ((purity - baseline) / baseline) * 100
                print(f"  Cs={row['Cs']:.1f}: {purity:.4f} ({change:+.1f}%)")
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
