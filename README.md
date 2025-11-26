# Feature Purity in Vision-Language Models

This repository contains code for investigating feature purity in CLIP-style vision-language models. Our work demonstrates that high-quality image-text alignment leads to more orthogonal (pure) feature representations, which in turn improves downstream generalization performance.

## 📋 Overview

We conduct three key experiments to validate the relationship between data quality, feature purity, and model generalization:

1. **CLIP vs LaCLIP Comparison**: Demonstrate that LaCLIP's improved captions lead to better feature purity
2. **Neuron Pruning Analysis**: Show that high-purity (orthogonal) neurons are more important for downstream tasks
3. **Misalignment Impact Study**: Investigate how controlled image-text misalignment reduces feature purity

## 🎯 Key Findings

- **High-purity neurons enhance generalization**: Retaining neurons with high feature purity (low cosine similarity) achieves better downstream performance
- **Data misalignment reduces feature purity**: Random text shuffling increases neuron cosine similarity, reducing orthogonality
- **LaCLIP outperforms CLIP**: Better captions lead to purer features and improved zero-shot classification

## 🚀 Getting Started

### Prerequisites

```bash
# Create conda environment
conda create -n feature-purity python=3.8
conda activate feature-purity

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

- PyTorch >= 1.11.0
- torchvision >= 0.12.0
- timm == 0.5.4
- open_clip_torch
- scikit-learn
- pandas
- tqdm
- numpy

### Dataset Preparation

#### CC3M Dataset
Download the CC3M dataset:
```bash
cd scripts
python download_cc3m.py
```

#### ImageNet (for evaluation)
Download ImageNet and organize it as:
```
data/imagenet/
├── train/
└── val/
```

## 📊 Experiments

### Experiment 1: CLIP vs LaCLIP Comparison

Compare zero-shot performance and feature quality between CLIP and LaCLIP models.

```bash
cd experiments/1_clip_vs_laclip
python batch_eval.py
```

**What it does:**
- Loads pretrained CLIP and LaCLIP checkpoints
- Evaluates zero-shot classification on multiple datasets (CIFAR10, CIFAR100, Food-101, etc.)
- Computes accuracy and Silhouette Score for feature quality assessment

**Key configurations:**
- Edit `base_ckpt_paths` in `batch_eval.py` to specify model checkpoints
- Results are saved to `{model_name}_full512_results/` directory

### Experiment 2: Neuron Pruning Analysis

Investigate the importance of feature purity by pruning neurons based on their orthogonality.

```bash
cd experiments/2_neuron_pruning
python run_eval_sam_sweep.py
```

**What it does:**
- Ranks 512 neurons by average pairwise absolute cosine similarity
- Evaluates three pruning strategies:
  - **Min Sparse**: Retain high-purity neurons (lowest similarity)
  - **Max Sparse**: Retain low-purity neurons (highest similarity)
  - **Full Features**: Use all 512 neurons (baseline)
- Sweeps across different numbers of retained neurons (200-500)

**Key configurations:**
- `ckpt_path`: Path to pretrained model checkpoint
- `DATASETS`: List of evaluation datasets
- `sam_values`: Number of neurons to retain

**Expected results:**
- High-purity neurons perform best
- Low-purity neurons perform worst
- Results saved as CSV files with accuracy and Silhouette Score

### Experiment 3: Misalignment Impact Study

Study how image-text misalignment affects feature purity through controlled text shuffling.

#### Step 3.1: Generate Misaligned Text

```bash
cd experiments/3_misalignment
python generate_misaligned_text.py
```

**What it does:**
- Creates similarity-based shuffled captions with different swap ratios (Cs = 0.1, 0.3, 0.5, 0.8)
- Uses multi-GPU TF-IDF + cosine similarity to find semantically similar pairs
- Swaps captions between similar image-text pairs to create controlled misalignment

**Key configurations:**
- `input_txt`: Original caption file
- `Cs_values`: List of swap probabilities
- `num_gpus`: Number of GPUs for parallel computation

**Output:**
- `cc3m_human_10w_Cs{XX}_similarity_simple.txt`: Shuffled captions for each Cs value

#### Step 3.2: Fine-tune with Misaligned Data

```bash
cd experiments/3_misalignment

# Fine-tune last linear layer only
torchrun --nproc_per_node=4 train.py \
    --train-data PATH/TO/SHUFFLED/CSV \
    --root PATH/TO/CC3M/IMAGES \
    --imagenet-root PATH/TO/IMAGENET \
    --output-dir results_Cs30 \
    --model CLIP_VITB16 \
    --lr 1e-3 --wd 0.5 \
    --warmup-epochs 1 --batch-size 256 --epochs 10
```

**What it does:**
- Loads pretrained CLIP model
- **Freezes all parameters except image_projection and text_projection**
- Fine-tunes only the final linear projection layers with misaligned data
- Evaluates feature purity by computing neuron cosine similarities

**Key configurations:**
The code automatically freezes all parameters except projections:
```python
for name, param in model.named_parameters():
    if "image_projection" not in name and "text_projection" not in name:
        param.requires_grad = False
```

**Expected results:**
- Higher Cs → Higher average absolute cosine similarity
- Higher Cs → Lower downstream task accuracy
- Feature purity decreases as misalignment increases

## 📈 Results Analysis

### Analyzing Neuron Cosine Similarity

After Experiment 3, analyze the feature purity:

```python
import torch
import numpy as np
from collections import OrderedDict
from src.models import CLIP_VITB16

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    model = CLIP_VITB16(rand_embed=False)
    model.load_state_dict(state_dict, strict=True)
    return model

def compute_avg_abs_cos_sim(projection):
    """Compute average absolute cosine similarity between neurons"""
    V = projection.T  # (512, 768)
    V = V / torch.norm(V, p=2, dim=1, keepdim=True)  # normalize
    cosine_sim = torch.matmul(V, V.T)  # (512, 512)
    cosine_sim = cosine_sim - torch.diag(torch.diag(cosine_sim))  # remove diagonal
    avg_abs_cos = torch.mean(torch.abs(cosine_sim))
    return avg_abs_cos.item()

# Load models with different Cs values
model = load_model("results_Cs30/checkpoint.pt")
text_proj = model.text_projection  # (768, 512)
purity = compute_avg_abs_cos_sim(text_proj)
print(f"Average absolute cosine similarity: {purity:.4f}")
```

## 📁 Repository Structure

```
feature-purity-clip/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/                              # Core source code
│   ├── models.py                     # CLIP model architectures
│   ├── losses.py                     # CLIP contrastive loss
│   ├── data.py                       # Dataset loaders
│   ├── utils.py                      # Utility functions
│   └── tokenizer.py                  # Text tokenizer
├── experiments/                      # Experimental scripts
│   ├── 1_clip_vs_laclip/
│   │   ├── README.md                 # Experiment 1 details
│   │   └── batch_eval.py             # Compare CLIP vs LaCLIP
│   ├── 2_neuron_pruning/
│   │   ├── README.md                 # Experiment 2 details
│   │   └── run_eval_sam_sweep.py     # Neuron pruning evaluation
│   └── 3_misalignment/
│       ├── README.md                 # Experiment 3 details
│       ├── generate_misaligned_text.py  # Create shuffled captions
│       └── train.py                  # Fine-tune with misalignment
├── scripts/                          # Utility scripts
│   ├── download_cc3m.py             # Download CC3M dataset
│   ├── eval_zeroshot_imagenet.py    # Zero-shot evaluation
│   └── eval_zeroshot_imagenet_laion.py
└── figures/                          # Generated figures (optional)
```

## 🔧 Model Checkpoints

### Required Checkpoints

Download pretrained models from [LaCLIP](https://github.com/LijieeFan/LaCLIP):

- **CC3M CLIP**: [Download](https://www.dropbox.com/s/5jsthdm85r2nfpz/cc3m_clip.pt?dl=0)
- **CC3M LaCLIP**: [Download](https://www.dropbox.com/s/k2e1tgsfmo0afme/cc3m_laclip.pt?dl=0)

Place checkpoints in a `checkpoints/` directory:
```
checkpoints/
├── cc3m_clip.pt
└── cc3m_laclip.pt
```

## 📊 Expected Results

### Experiment 1: CLIP vs LaCLIP Comparison

Comparison of CLIP and LaCLIP on Accuracy (%) and Silhouette Score (SS).

| Model | Food-101 Acc | Food-101 SS | CIFAR-10 Acc | CIFAR-10 SS | Caltech-101 Acc | Caltech-101 SS | CIFAR-100 Acc | CIFAR-100 SS | Pets Acc | Pets SS | STL-10 Acc | STL-10 SS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CC12M CLIP** | 50.8 | 0.034 | 64.9 | 0.113 | 77.4 | 0.225 | 38.5 | 0.005 | 64.1 | 0.069 | 91.0 | 0.195 |
| **CC12M LaCLIP** | **60.7** | **0.038** | **75.1** | **0.157** | **83.3** | **0.276** | **43.9** | **0.029** | **72.4** | **0.070** | **95.1** | **0.273** |
| **RedCaps CLIP** | 81.5 | 0.125 | 70.4 | 0.100 | 72.8 | 0.210 | 39.9 | -0.002 | **82.7** | **0.091** | **92.8** | 0.226 |
| **RedCaps LaCLIP** | **85.0** | **0.175** | **74.8** | **0.107** | **76.4** | **0.233** | **40.7** | **0.011** | 78.2 | 0.074 | 91.4 | **0.275** |
| **LAION CLIP** | 85.5 | 0.116 | 93.0 | 0.181 | 91.2 | 0.258 | 71.7 | 0.078 | 90.1 | 0.122 | 97.3 | 0.223 |
| **LAION LaCLIP** | **86.5** | **0.148** | **93.5** | **0.215** | **92.4** | **0.306** | **73.9** | **0.108** | **90.9** | **0.152** | **98.4** | **0.260** |

### Experiment 2: Neuron Pruning

With 450 neurons retained:
- **Min Sparse (High Purity)**: ~42% accuracy
- **Random**: ~40% accuracy
- **Max Sparse (Low Purity)**: ~35% accuracy

### Experiment 3: Misalignment Impact

| Cs | Avg Abs Cos Sim | Accuracy Drop |
|----|-----------------|---------------|
| 0.0 | 0.12 | 0% |
| 0.1 | 0.14 | -2% |
| 0.3 | 0.17 | -5% |
| 0.5 | 0.21 | -8% |
| 0.8 | 0.26 | -12% |

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{sun2025contrastive,
  title={Contrastive Learning with Data Misalignment: Feature Purity, Training Dynamics and Theoretical Generalization Guarantees},
  author={Sun, Jiawei and Zhang, Shuai and Li, Hongkang and Wang, Meng},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## 🙏 Acknowledgements

This work builds upon [LaCLIP](https://github.com/LijieeFan/LaCLIP):

```bibtex
@inproceedings{fan2023improving,
  title={Improving CLIP Training with Language Rewrites},
  author={Fan, Lijie and Krishnan, Dilip and Isola, Phillip and Katabi, Dina and Tian, Yonglong},
  booktitle={NeurIPS},
  year={2023}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
