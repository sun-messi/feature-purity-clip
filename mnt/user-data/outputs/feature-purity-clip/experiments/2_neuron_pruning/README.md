# Experiment 2: Neuron Pruning Analysis

## Overview

This experiment investigates the relationship between feature purity and model performance by selectively pruning neurons in the final linear projection layer. We demonstrate that neurons with high feature purity (low cosine similarity with other neurons) are more critical for downstream generalization.

## What This Experiment Does

1. **Analyzes Feature Purity**: Computes pairwise cosine similarity between all 512 neurons in the final projection layer
2. **Ranks Neurons**: Orders neurons from most orthogonal (pure) to least orthogonal (mixed)
3. **Selective Pruning**: Evaluates three pruning strategies:
   - **Min Sparse**: Keep high-purity neurons (lowest cosine similarity)
   - **Max Sparse**: Keep low-purity neurons (highest cosine similarity)
   - **Full Features**: Baseline with all 512 neurons
4. **Performance Evaluation**: Measures zero-shot accuracy and Silhouette Score on downstream tasks

## Key Concepts

### Feature Purity

Feature purity measures how orthogonal (independent) a neuron's representation is from other neurons:

```
Purity(neuron_j) = 1 / avg_i(|cos_sim(neuron_j, neuron_i)|)
```

- **High purity** = Low average cosine similarity = More orthogonal = Better
- **Low purity** = High average cosine similarity = More mixed = Worse

### Neuron Selection Strategies

```python
# Compute cosine similarities
image_VVT = torch.matmul(image_proj, image_proj.T)  # (512, 512)
text_VVT = torch.matmul(text_proj, text_proj.T)      # (512, 512)

# Remove diagonal (self-similarity)
image_VVT -= torch.diag(torch.diag(image_VVT))
text_VVT -= torch.diag(torch.diag(text_VVT))

# Compute total similarity for each neuron
total_row_sum = torch.sum(torch.abs(image_VVT), dim=1) + torch.sum(torch.abs(text_VVT), dim=1)

# Min Sparse: Select neurons with lowest similarity (highest purity)
min_indices = torch.topk(total_row_sum, sam, largest=False)

# Max Sparse: Select neurons with highest similarity (lowest purity)
max_indices = torch.topk(total_row_sum, sam, largest=True)
```

## Running the Experiment

### Basic Usage

```bash
cd experiments/2_neuron_pruning
python run_eval_sam_sweep.py
```

### Configuration

Edit key parameters in `run_eval_sam_sweep.py`:

```python
# Model checkpoint
ckpt_path = "/path/to/cc3m_laclip.pt"

# Evaluation datasets
DATASETS = ["CIFAR10", "CIFAR100", "FOOD101", "CALTECH101", "PETS", "DTD", "EUROSAT"]

# Number of neurons to retain (sweep)
sam_values = list(range(200, 501, 50))  # [200, 250, 300, ..., 500]

# GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### Advanced Options

```python
# Adjust batch size if needed
batch_size = 512

# Change evaluation datasets
DATASETS = ["CIFAR10", "CIFAR100"]  # Test on fewer datasets first

# Fine-grained neuron sweep
sam_values = list(range(100, 512, 20))  # More granular analysis
```

## Output

Results are saved to `save_dir_base`:

```
result_cc3m_laclip/
├── cifar10_laclip_accuracy_vs_sam.csv
├── cifar10_laclip_silhouette_vs_sam.csv
├── cifar100_laclip_accuracy_vs_sam.csv
├── cifar100_laclip_silhouette_vs_sam.csv
├── food101_laclip_accuracy_vs_sam.csv
└── food101_laclip_silhouette_vs_sam.csv
```

### CSV Format

Each accuracy CSV contains:
```
sam,Model,Full Features,Min Sparse Features,Max Sparse Features
200,CC3M LaCLIP,0.458,0.423,0.385
250,CC3M LaCLIP,0.458,0.438,0.401
...
```

Each silhouette CSV contains:
```
sam,Model,Full Features,Min Sparse Features,Max Sparse Features
200,CC3M LaCLIP,0.152,0.148,0.138
250,CC3M LaCLIP,0.152,0.151,0.142
...
```

## Expected Results

### Zero-shot Accuracy on CIFAR-10

| Neurons | Full (512) | Min Sparse | Max Sparse | Δ (Min - Max) |
|---------|------------|------------|------------|---------------|
| 200 | 45.8% | 42.3% | 38.5% | **+3.8%** |
| 250 | 45.8% | 43.8% | 40.1% | **+3.7%** |
| 300 | 45.8% | 44.5% | 41.2% | **+3.3%** |
| 350 | 45.8% | 45.0% | 42.5% | **+2.5%** |
| 400 | 45.8% | 45.3% | 43.8% | **+1.5%** |
| 450 | 45.8% | 45.6% | 44.9% | **+0.7%** |
| 500 | 45.8% | 45.7% | 45.5% | **+0.2%** |

### Key Observations

1. **Min Sparse > Max Sparse**: High-purity neurons consistently outperform low-purity neurons
2. **Performance Gap**: The gap is largest when using fewer neurons (200-300)
3. **Convergence**: With more neurons (450+), all strategies approach baseline performance

### Silhouette Score Analysis

Similar patterns emerge for feature quality:
- Min Sparse features maintain better cluster separation
- Max Sparse features show degraded feature quality
- The gap narrows as more neurons are included

## Visualization

### Plot Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("result_cc3m_laclip/cifar10_laclip_accuracy_vs_sam.csv")

plt.figure(figsize=(10, 6))
plt.plot(df['sam'], df['Full Features'], 'k--', label='Full (512)', linewidth=2)
plt.plot(df['sam'], df['Min Sparse Features'], 'b-o', label='Min Sparse (High Purity)', linewidth=2)
plt.plot(df['sam'], df['Max Sparse Features'], 'r-s', label='Max Sparse (Low Purity)', linewidth=2)

plt.xlabel('Number of Retained Neurons', fontsize=12)
plt.ylabel('Zero-shot Accuracy (%)', fontsize=12)
plt.title('CIFAR-10: Effect of Neuron Pruning on Accuracy', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('neuron_pruning_cifar10.png', dpi=300)
plt.show()
```

### Generate Comparison Plot

```python
import numpy as np

# Calculate performance gap
df['Gap'] = df['Min Sparse Features'] - df['Max Sparse Features']

plt.figure(figsize=(10, 6))
plt.plot(df['sam'], df['Gap'], 'g-^', linewidth=2, markersize=8)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Number of Retained Neurons', fontsize=12)
plt.ylabel('Accuracy Gap (Min - Max) %', fontsize=12)
plt.title('Performance Gap: High-Purity vs Low-Purity Neurons', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('purity_gap.png', dpi=300)
plt.show()
```

## Understanding the Code

### Core Functions

```python
def normalize_tensor(V):
    """Normalize neuron vectors to unit length"""
    norm = torch.norm(V, p=2, dim=1, keepdim=True)
    return V / norm

def find_min_off_diagonal_sum(image_proj, text_proj, sam=450):
    """Find neurons with lowest total cosine similarity (highest purity)"""
    # 1. Compute V @ V^T for both image and text projections
    # 2. Remove diagonal elements (self-similarity)
    # 3. Sum absolute similarities for each neuron
    # 4. Select top sam neurons with lowest sums
    
def find_max_off_diagonal_sum(image_proj, text_proj, sam=450):
    """Find neurons with highest total cosine similarity (lowest purity)"""
    # Similar to find_min, but selects highest sums
```

### Evaluation Pipeline

```python
def evaluate_zeroshot_classification(model, dataset, cls_names, sam=450):
    # 1. Extract full 512-dimensional image features
    # 2. Extract text features for all class prompts
    # 3. Load projection matrices and rank neurons by purity
    # 4. Create sparse features:
    #    - Min Sparse: Zero out all except high-purity neurons
    #    - Max Sparse: Zero out all except low-purity neurons
    # 5. Evaluate accuracy and Silhouette Score for each variant
```

## Analysis and Insights

### Why High-Purity Neurons Matter

1. **Orthogonality**: High-purity neurons encode independent features
2. **Specialization**: Each neuron captures distinct semantic information
3. **Robustness**: Independent features are less prone to interference
4. **Generalization**: Orthogonal representations transfer better to new tasks

### Why Low-Purity Neurons Underperform

1. **Redundancy**: Similar neurons encode overlapping information
2. **Interference**: Mixed features create confusion in classification
3. **Fragility**: Dependent features are sensitive to distribution shifts
4. **Inefficiency**: Multiple neurons needed to encode what one could capture

## Troubleshooting

### Common Issues

1. **Memory Error**:
   ```python
   # Reduce batch size
   batch_size = 256  # Instead of 512
   ```

2. **CUDA Out of Memory**:
   ```python
   # Process datasets sequentially
   DATASETS = ["CIFAR10"]  # Test one at a time
   ```

3. **Slow Evaluation**:
   ```python
   # Reduce number of sweep points
   sam_values = [200, 300, 400, 500]  # Fewer points
   ```

## Extensions

### Experiment Variations

1. **Different Models**: Test with various CLIP architectures (ViT-L/14, etc.)
2. **Layer Analysis**: Examine purity at different network depths
3. **Training Dynamics**: Track purity evolution during training
4. **Other Metrics**: Explore different purity measures (L1, L∞, etc.)

### Advanced Analysis

```python
# Analyze individual neuron importance
def analyze_neuron_importance(model, dataset):
    """Rank neurons by individual impact on accuracy"""
    baseline_acc = evaluate_full(model, dataset)
    
    neuron_importance = []
    for i in range(512):
        # Ablate neuron i
        pruned_acc = evaluate_with_ablation(model, dataset, ablate_idx=i)
        importance = baseline_acc - pruned_acc
        neuron_importance.append((i, importance))
    
    return sorted(neuron_importance, key=lambda x: x[1], reverse=True)
```

## Next Steps

1. **Run Experiment**: Execute the sweep across all datasets
2. **Analyze Results**: Generate plots and statistical analysis
3. **Compare Models**: Test on both CLIP and LaCLIP checkpoints
4. **Move to Experiment 3**: Investigate misalignment effects

## References

- [Neural Network Interpretability](https://distill.pub/2020/circuits/)
- [Feature Visualization](https://distill.pub/2017/feature-visualization/)
- [Sparse Neural Networks](https://arxiv.org/abs/2102.00554)
