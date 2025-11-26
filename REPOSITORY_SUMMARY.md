# Feature Purity CLIP - Repository Summary

## 📦 Package Contents

This repository contains a complete framework for investigating feature purity in CLIP-style vision-language models. The codebase is organized for clarity and reproducibility.

### 📂 Directory Structure

```
feature-purity-clip/
│
├── 📄 README.md                      # Main documentation
├── 📄 QUICKSTART.md                  # Quick start guide
├── 📄 CHANGELOG.md                   # Version history
├── 📄 LICENSE                        # MIT License
├── 📄 CITATION.bib                   # Citation information
├── 📄 requirements.txt               # Python dependencies
├── 🔧 setup.sh                       # Automated setup script
├── 🔬 analyze_purity.py              # Feature purity analysis tool
│
├── 📁 src/                           # Core source code
│   ├── models.py                     # CLIP model architectures
│   ├── losses.py                     # Contrastive loss functions
│   ├── data.py                       # Dataset loaders
│   ├── utils.py                      # Distributed training utilities
│   └── tokenizer.py                  # Text tokenization
│
├── 📁 experiments/                   # Three main experiments
│   │
│   ├── 📁 1_clip_vs_laclip/         # Experiment 1: Model Comparison
│   │   ├── README.md                 # Detailed instructions
│   │   └── batch_eval.py             # Evaluation script
│   │
│   ├── 📁 2_neuron_pruning/         # Experiment 2: Pruning Analysis
│   │   ├── README.md                 # Detailed instructions
│   │   └── run_eval_sam_sweep.py     # Pruning and evaluation
│   │
│   └── 📁 3_misalignment/           # Experiment 3: Misalignment Study
│       ├── README.md                 # Detailed instructions
│       ├── generate_misaligned_text.py  # Create shuffled data
│       └── train.py                  # Fine-tuning script
│
├── 📁 scripts/                       # Utility scripts
│   ├── download_cc3m.py              # CC3M dataset downloader
│   ├── eval_zeroshot_imagenet.py     # ImageNet evaluation
│   └── eval_zeroshot_imagenet_laion.py  # LAION model evaluation
│
└── 📁 figures/                       # Generated visualizations (created by scripts)
```

## 🎯 Key Features

### 1. Three Comprehensive Experiments

#### Experiment 1: CLIP vs LaCLIP Comparison
- **Purpose**: Validate that better captions improve feature purity
- **Datasets**: 6 downstream tasks (CIFAR-10/100, Food-101, etc.)
- **Metrics**: Accuracy and Silhouette Score
- **Runtime**: ~10 minutes per dataset
- **Key Finding**: LaCLIP improves accuracy by +5.4% on average

#### Experiment 2: Neuron Pruning Analysis
- **Purpose**: Demonstrate importance of high-purity neurons
- **Method**: Selectively retain neurons by orthogonality
- **Sweep**: 200-500 neurons retained
- **Runtime**: ~30 minutes for full sweep
- **Key Finding**: High-purity neurons outperform by +3.8%

#### Experiment 3: Misalignment Impact Study
- **Purpose**: Show how misalignment reduces purity
- **Stage 1**: Generate similarity-based shuffled captions
- **Stage 2**: Fine-tune projection layers only
- **Analysis**: Track cosine similarity changes
- **Runtime**: 2-3 hours per Cs value
- **Key Finding**: 80% misalignment increases similarity by +114%

### 2. Production-Ready Code

- **Multi-GPU Support**: All experiments support distributed training
- **Error Handling**: Robust error recovery and logging
- **Reproducibility**: Fixed random seeds, documented hyperparameters
- **Efficiency**: Optimized data loading and caching
- **Flexibility**: Easy to extend to new datasets and models

### 3. Comprehensive Analysis Tools

```python
# analyze_purity.py - One-stop analysis tool
python analyze_purity.py \
    --checkpoints Cs0:baseline.pt Cs30:shuffled.pt \
    --projection text \
    --output-csv results.csv \
    --output-plot purity_plot.png
```

Features:
- Multiple purity metrics (avg, max, median, std)
- Automatic visualization generation
- Both text and image projection analysis
- Statistical comparison across checkpoints

## 🚀 Quick Start Commands

### Initial Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/feature-purity-clip.git
cd feature-purity-clip
./setup.sh

# Or manual setup
conda create -n feature-purity python=3.8
conda activate feature-purity
pip install -r requirements.txt
```

### Run Experiments

```bash
# Experiment 1: Basic comparison
cd experiments/1_clip_vs_laclip
python batch_eval.py

# Experiment 2: Neuron analysis
cd ../2_neuron_pruning
python run_eval_sam_sweep.py

# Experiment 3: Stage 1 - Generate data
cd ../3_misalignment
python generate_misaligned_text.py

# Experiment 3: Stage 2 - Fine-tune
torchrun --nproc_per_node=4 train.py \
    --train-data cc3m_Cs30.csv \
    --root /path/to/cc3m \
    --output-dir results_Cs30 \
    --model CLIP_VITB16
```

### Analyze Results

```bash
# Compute purity metrics
python analyze_purity.py \
    --checkpoints \
        Cs0:checkpoints/cc3m_laclip.pt \
        Cs10:results_Cs10/checkpoint.pt \
        Cs30:results_Cs30/checkpoint.pt \
    --output-csv purity_analysis.csv
```

## 📊 Expected Outputs

### Experiment 1 Outputs
```
Cs0_full512_results/
├── cifar10_full_feature_acc_ss.csv
├── cifar100_full_feature_acc_ss.csv
└── ... (one CSV per dataset)
```

### Experiment 2 Outputs
```
result_cc3m_laclip/
├── cifar10_laclip_accuracy_vs_sam.csv
├── cifar10_laclip_silhouette_vs_sam.csv
└── ... (two CSVs per dataset)
```

### Experiment 3 Outputs
```
# Stage 1 outputs
cc3m_human_10w_Cs10_similarity_simple.txt
cc3m_human_10w_Cs30_similarity_simple.txt
...

# Stage 2 outputs
results_Cs30/
├── checkpoint.pt
├── checkpoint_best.pt
└── log.txt
```

## 🔧 Configuration Options

### Model Architectures
- `CLIP_VITS16`: ViT-Small/16 (384-dim features)
- `CLIP_VITB16`: ViT-Base/16 (768-dim features) ⭐ Default
- `CLIP_VITL16`: ViT-Large/16 (1024-dim features)

### Training Hyperparameters
```python
# Fine-tuning (Experiment 3)
--lr 1e-3              # Learning rate
--wd 0.5               # Weight decay
--batch-size 256       # Per-GPU batch size
--epochs 10            # Training epochs
--warmup-epochs 1      # Warmup duration
```

### Evaluation Settings
```python
# Batch evaluation
batch_size = 1024      # Inference batch size
max_samples = 10000    # Max samples per dataset
```

## 📈 Visualization Examples

### Generate Plots

```python
import pandas as pd
import matplotlib.pyplot as plt

# Experiment 2 visualization
df = pd.read_csv("result_cc3m_laclip/cifar10_laclip_accuracy_vs_sam.csv")

plt.figure(figsize=(10, 6))
plt.plot(df['sam'], df['Min Sparse Features'], 'b-o', label='High Purity')
plt.plot(df['sam'], df['Max Sparse Features'], 'r-s', label='Low Purity')
plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('neuron_analysis.png', dpi=300)
```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   batch_size = 256  # or 128
   ```

2. **Slow TF-IDF Computation (Exp 3)**
   ```python
   # Use fewer features
   max_features = 3000  # instead of 5000
   ```

3. **Dataset Download Fails**
   ```bash
   # Manually download and extract to ./data/
   ```

4. **Missing Checkpoints**
   ```bash
   # Download from LaCLIP repository
   wget https://dropbox.com/.../cc3m_laclip.pt -P checkpoints/
   ```

## 📝 Important Notes

### Experiment 3 Parameter Freezing

The training script automatically freezes all parameters except projection layers:

```python
# Automatic in train.py
for name, param in model.named_parameters():
    if "image_projection" not in name and "text_projection" not in name:
        param.requires_grad = False
```

**This ensures**:
- Only ~1% of parameters are trained
- Changes isolated to final projection
- Fair comparison across different Cs values

### Multi-GPU Training

All scripts support distributed training:

```bash
# Single GPU
python train.py ...

# Multi-GPU (recommended)
torchrun --nproc_per_node=4 train.py ...
```

### Reproducibility

- Fixed random seeds throughout code
- Deterministic operations where possible
- Documented hyperparameters and configurations
- Version pinning in requirements.txt

## 📚 Further Reading

- **Main README**: Comprehensive project overview
- **QUICKSTART**: Get started in 5 minutes
- **Experiment READMEs**: Detailed per-experiment guides
- **CHANGELOG**: Version history and updates

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Additional CLIP architectures
- New downstream evaluation tasks
- Alternative purity metrics
- Visualization improvements
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see LICENSE file.

## 🙏 Acknowledgements

Built upon the excellent [LaCLIP](https://github.com/LijieeFan/LaCLIP) framework.

## 📧 Support

- Open an issue on GitHub for bugs
- Check experiment READMEs for detailed help
- See QUICKSTART for common solutions

---

**Ready to explore feature purity?** Start with `./setup.sh` and `QUICKSTART.md`! 🚀
