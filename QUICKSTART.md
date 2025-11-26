# Quick Start Guide

Get up and running with feature purity experiments in 5 minutes!

## Prerequisites

- Linux system with CUDA-capable GPU(s)
- Python 3.8+
- 16GB+ GPU memory recommended

## Setup (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/feature-purity-clip.git
cd feature-purity-clip

# 2. Create conda environment
conda create -n feature-purity python=3.8 -y
conda activate feature-purity

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pretrained models
mkdir -p checkpoints
cd checkpoints
wget https://www.dropbox.com/s/5jsthdm85r2nfpz/cc3m_clip.pt
wget https://www.dropbox.com/s/k2e1tgsfmo0afme/cc3m_laclip.pt
cd ..
```

## Run Your First Experiment (2 minutes)

### Experiment 1: CLIP vs LaCLIP

```bash
cd experiments/1_clip_vs_laclip

# Edit batch_eval.py to set checkpoint paths
# Then run:
python batch_eval.py
```

**Expected runtime**: ~10 minutes on single GPU  
**Output**: CSV files with accuracy and Silhouette Score for each dataset

### Experiment 2: Neuron Pruning

```bash
cd experiments/2_neuron_pruning

# Edit run_eval_sam_sweep.py to set checkpoint path
# Then run:
python run_eval_sam_sweep.py
```

**Expected runtime**: ~30 minutes for full sweep  
**Output**: Accuracy vs number of neurons for different pruning strategies

### Experiment 3: Misalignment (Advanced)

This requires more setup and compute. See `experiments/3_misalignment/README.md` for details.

## Understanding the Output

### Experiment 1 Output

```
Cs0_full512_results/
├── cifar10_full_feature_acc_ss.csv
└── ...
```

Each CSV contains:
- `Accuracy`: Classification accuracy (0-1)
- `Silhouette Score`: Feature quality (-1 to 1, higher is better)

### Experiment 2 Output

```
result_cc3m_laclip/
├── cifar10_laclip_accuracy_vs_sam.csv
└── ...
```

Each CSV shows performance with different numbers of neurons retained.

## Quick Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load and plot Experiment 1 results
df = pd.read_csv("Cs0_full512_results/cifar10_full_feature_acc_ss.csv")
print(f"CIFAR-10 Accuracy: {df['Accuracy'][0]:.2%}")
print(f"Silhouette Score: {df['Silhouette Score'][0]:.3f}")

# Load and plot Experiment 2 results
df = pd.read_csv("result_cc3m_laclip/cifar10_laclip_accuracy_vs_sam.csv")
plt.plot(df['sam'], df['Min Sparse Features'], label='High Purity')
plt.plot(df['sam'], df['Max Sparse Features'], label='Low Purity')
plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('neuron_pruning.png')
```

## Common Issues

### CUDA Out of Memory
```python
# Reduce batch size in the script
batch_size = 256  # Instead of 512 or 1024
```

### Missing Checkpoints
```bash
# Manually download from LaCLIP repository
wget https://www.dropbox.com/s/k2e1tgsfmo0afme/cc3m_laclip.pt -P checkpoints/
```

### Dataset Download Fails
```python
# Datasets auto-download to ./data/
# If download fails, check internet connection or download manually
```

## Next Steps

1. **Analyze Results**: Use the visualization scripts in each experiment README
2. **Try Different Models**: Test with other CLIP variants
3. **Run Full Experiment 3**: Generate misaligned data and fine-tune

## Getting Help

- Check individual experiment READMEs for detailed instructions
- Open an issue on GitHub for bugs or questions
- See main README.md for comprehensive documentation

## Minimal Working Example

```python
# minimal_test.py - Verify installation
import torch
from src.models import CLIP_VITB16
from collections import OrderedDict

# Load model
ckpt = torch.load("checkpoints/cc3m_laclip.pt", map_location='cpu')
state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    state_dict[k.replace('module.', '')] = v

model = CLIP_VITB16(rand_embed=False)
model.load_state_dict(state_dict, strict=True)

# Check model
print("✓ Model loaded successfully!")
print(f"  Image projection: {model.image_projection.shape}")
print(f"  Text projection: {model.text_projection.shape}")

# Compute feature purity
text_proj = model.text_projection.T  # (512, 768)
V = text_proj / torch.norm(text_proj, p=2, dim=1, keepdim=True)
cos_sim = torch.matmul(V, V.T)
cos_sim -= torch.diag(torch.diag(cos_sim))
purity = torch.mean(torch.abs(cos_sim)).item()

print(f"  Avg abs cosine similarity: {purity:.4f}")
print("\n✓ Everything working correctly!")
```

Run this to verify your setup:
```bash
python minimal_test.py
```

Expected output:
```
✓ Model loaded successfully!
  Image projection: torch.Size([768, 512])
  Text projection: torch.Size([512, 512])
  Avg abs cosine similarity: 0.1234
✓ Everything working correctly!
```

Happy experimenting! 🚀
