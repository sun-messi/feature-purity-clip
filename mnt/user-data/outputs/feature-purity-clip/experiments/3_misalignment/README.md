# Experiment 3: Misalignment Impact Study

## Overview

This experiment investigates how image-text misalignment affects feature purity in CLIP models. We create controlled misalignment by swapping captions between semantically similar image-text pairs, then fine-tune only the final projection layers to observe changes in neuron orthogonality.

## Hypothesis

**As image-text misalignment increases → Feature purity decreases → Downstream performance degrades**

We test this by:
1. Creating datasets with varying degrees of caption shuffling (Cs = 0.1, 0.3, 0.5, 0.8)
2. Fine-tuning the projection layers with misaligned data
3. Measuring changes in average neuron cosine similarity
4. Evaluating downstream task performance

## Two-Stage Process

### Stage 1: Generate Misaligned Text
Create similarity-based shuffled captions using multi-GPU TF-IDF analysis.

### Stage 2: Fine-tune with Misalignment
Train only the final projection layers on misaligned data and analyze feature purity.

---

## Stage 1: Generate Misaligned Text

### What It Does

1. **Compute Text Similarity**: Uses TF-IDF + cosine similarity to find semantically similar caption pairs
2. **Multi-GPU Acceleration**: Distributes computation across GPUs for faster processing
3. **Similarity-Based Swapping**: Swaps captions between similar pairs (not random)
4. **Controlled Misalignment**: Creates multiple datasets with different swap ratios (Cs)

### Why Similarity-Based?

Unlike random shuffling, swapping similar captions creates more subtle misalignment:
- **Random swap**: "dog playing" ↔ "skyscraper downtown" (obvious mismatch)
- **Similarity-based**: "dog playing" ↔ "puppy running" (subtle mismatch)

This better models real-world caption noise and is harder for the model to ignore.

### Running Stage 1

```bash
cd experiments/3_misalignment
python generate_misaligned_text.py
```

### Configuration

```python
# Input
input_txt = "cc3m_human_10w.txt"  # Original captions (one per line)

# Swap ratios to generate
Cs_values = [0.1, 0.3, 0.5, 0.8]  # 10%, 30%, 50%, 80% swapped

# Similarity parameters
top_k = 10  # Consider top-10 most similar candidates
num_gpus = 8  # Use 8 GPUs for parallel computation
batch_size = 1000  # Process 1000 texts per batch
```

### How It Works

```python
# 1. Preprocess Text
texts = ["a dog playing in park", "a puppy running outdoors", ...]
processed = preprocess_texts_parallel(texts)  # Clean and normalize

# 2. Compute TF-IDF Vectors
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(processed)  # (N, 5000)

# 3. Multi-GPU Similarity Computation
similarity_matrix = compute_similarity_matrix_multigpu(texts, calculator)
# similarity_matrix[i,j] = cosine_similarity(text_i, text_j)

# 4. Find Similar Pairs and Swap
for source_idx in range(N):
    if swapped_count < target_swap_count:
        # Find top-k most similar captions
        candidates = find_similar_candidates(similarity_matrix, source_idx, top_k)
        # Randomly pick one to swap with
        target_idx = random.choice(candidates)
        # Swap captions
        captions[source_idx], captions[target_idx] = captions[target_idx], captions[source_idx]
```

### Output Files

```
cc3m_human_10w_Cs10_similarity_simple.txt   # 10% swapped
cc3m_human_10w_Cs30_similarity_simple.txt   # 30% swapped
cc3m_human_10w_Cs50_similarity_simple.txt   # 50% swapped
cc3m_human_10w_Cs80_similarity_simple.txt   # 80% swapped
```

Each file contains the same number of lines as input, with specified percentage of captions swapped.

### Verification

The script prints swap examples:
```
交换示例 (Cs=0.3):
交换对 1 (相似度: 0.847):
  位置 42: a puppy running in a grassy field...
  位置 158: a dog playing fetch outdoors...
交换对 2 (相似度: 0.823):
  位置 97: sunset over the ocean with orange sky...
  位置 234: beautiful orange sunset at the beach...
```

### Troubleshooting Stage 1

1. **Out of GPU Memory**:
   ```python
   batch_size = 500  # Reduce batch size
   num_gpus = 4      # Use fewer GPUs
   ```

2. **Slow Processing**:
   ```python
   max_features = 3000  # Reduce TF-IDF features
   ```

3. **Low-Quality Matches**:
   ```python
   top_k = 20  # Increase candidate pool
   min_df = 5  # Adjust TF-IDF parameters
   ```

---

## Stage 2: Fine-tune with Misalignment

### What It Does

1. **Freeze Most Parameters**: Only train image_projection and text_projection layers
2. **Fine-tune on Misaligned Data**: Use shuffled captions from Stage 1
3. **Monitor Feature Purity**: Track neuron cosine similarity during training
4. **Evaluate Performance**: Measure downstream task accuracy

### Why Freeze Most Layers?

- **Isolate Effect**: Changes in purity come only from projection layers
- **Faster Training**: Only ~1% of parameters are trainable
- **Fair Comparison**: All models start from same pretrained backbone

### Running Stage 2

```bash
cd experiments/3_misalignment

# Example: Train with 30% misalignment
torchrun --nproc_per_node=4 train.py \
    --train-data cc3m_Cs30.csv \
    --root /path/to/cc3m/images \
    --imagenet-root /path/to/imagenet \
    --output-dir results_Cs30 \
    --model CLIP_VITB16 \
    --lr 1e-3 --wd 0.5 \
    --warmup-epochs 1 --batch-size 256 --epochs 10
```

### Configuration

#### Data Preparation

First, create CSV file with shuffled captions:
```python
import pandas as pd

# Original CSV format: image_path,caption
df_original = pd.read_csv("cc3m_original.csv", header=None)
image_paths = df_original[0].tolist()

# Load shuffled captions
with open("cc3m_human_10w_Cs30_similarity_simple.txt") as f:
    shuffled_captions = [line.strip() for line in f]

# Create new CSV
df_shuffled = pd.DataFrame({
    0: image_paths,
    1: shuffled_captions
})
df_shuffled.to_csv("cc3m_Cs30.csv", header=False, index=False)
```

#### Training Parameters

```bash
# Distributed training
--nproc_per_node=4        # Number of GPUs

# Data
--train-data cc3m_Cs30.csv                    # Shuffled data
--root /path/to/cc3m/images                   # Image directory
--imagenet-root /path/to/imagenet             # For validation

# Model
--model CLIP_VITB16                           # Architecture
--output-dir results_Cs30                     # Save directory

# Optimization
--lr 1e-3                                     # Learning rate
--wd 0.5                                      # Weight decay
--warmup-epochs 1                             # Warmup period
--batch-size 256                              # Per-GPU batch size
--epochs 10                                   # Total epochs
```

### Automatic Parameter Freezing

The code automatically freezes all parameters except projections:

```python
# In train.py
for name, param in model.named_parameters():
    if "image_projection" not in name and "text_projection" not in name:
        param.requires_grad = False
    else:
        print(f"Training parameter: {name}")

# Output:
# Training parameter: image_projection
# Training parameter: text_projection
```

### Training Output

```
Epoch: [0]  [  0/3906]  lr: 0.000100  loss: 4.1234  clip_loss: 4.1234  clip_acc: 15.20
Epoch: [0]  [100/3906]  lr: 0.000325  loss: 3.8567  clip_loss: 3.8567  clip_acc: 22.80
...
ImageNet zero-shot accuracy: 43.2%
Saved checkpoint to results_Cs30/checkpoint.pt
```

### Output Files

```
results_Cs30/
├── checkpoint.pt           # Final model weights
├── checkpoint_best.pt      # Best validation checkpoint
└── log.txt                 # Training logs
```

---

## Analyzing Feature Purity

After training, analyze how misalignment affected neuron orthogonality.

### Compute Average Cosine Similarity

```python
import torch
import numpy as np
from collections import OrderedDict
from src.models import CLIP_VITB16

def normalize_tensor(V):
    """Normalize vectors to unit length"""
    norm = torch.norm(V, p=2, dim=1, keepdim=True)
    return V / norm

def compute_avg_abs_cos_sim(V):
    """Compute average absolute cosine similarity"""
    V = normalize_tensor(V)  # (512, 768) normalized
    cosine_sim = torch.matmul(V, V.T)  # (512, 512)
    
    # Remove diagonal (self-similarity = 1)
    cosine_sim = cosine_sim - torch.diag(torch.diag(cosine_sim))
    
    # Average of absolute values
    avg_abs_cos = torch.mean(torch.abs(cosine_sim))
    return avg_abs_cos.item()

# Load model
ckpt = torch.load("results_Cs30/checkpoint.pt", map_location='cpu')
state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    state_dict[k.replace('module.', '')] = v

model = CLIP_VITB16(rand_embed=False)
model.load_state_dict(state_dict, strict=True)

# Analyze text projection
text_proj = model.text_projection  # (768, 512)
image_proj = model.image_projection  # (768, 512)

text_purity = compute_avg_abs_cos_sim(text_proj.T)
image_purity = compute_avg_abs_cos_sim(image_proj.T)

print(f"Text Projection - Avg Abs Cos Sim: {text_purity:.4f}")
print(f"Image Projection - Avg Abs Cos Sim: {image_purity:.4f}")
```

### Batch Analysis Across All Cs Values

```python
checkpoints = {
    "Cs=0": "finetune_result_CLIP/checkpoint.pt",
    "Cs=0.1": "finetune_result_Cs10_similarity/checkpoint.pt",
    "Cs=0.3": "finetune_result_Cs30_similarity/checkpoint.pt",
    "Cs=0.5": "finetune_result_Cs50_similarity/checkpoint.pt",
    "Cs=0.8": "finetune_result_Cs80_similarity/checkpoint.pt",
}

results = []
for label, ckpt_path in checkpoints.items():
    model = load_model(ckpt_path)
    text_proj = model.text_projection.T  # (512, 768)
    purity = compute_avg_abs_cos_sim(text_proj)
    
    Cs = float(label.split('=')[1])
    results.append({'Cs': Cs, 'Purity': purity})
    print(f"{label}: Avg Abs Cos Sim = {purity:.4f}")

# Save results
df = pd.DataFrame(results)
df.to_csv("purity_vs_misalignment.csv", index=False)
```

### Visualize Purity vs Misalignment

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(df['Cs'], df['Purity'], 'b-o', linewidth=2, markersize=10)
plt.xlabel('Shuffling Probability (Cs)', fontsize=14)
plt.ylabel('Avg Absolute Cosine Similarity', fontsize=14)
plt.title('Feature Purity Decreases with Misalignment', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('purity_vs_cs.png', dpi=300)
```

### Histogram of Cosine Similarities

```python
def plot_cosine_histogram(model, Cs_value):
    """Plot distribution of neuron cosine similarities"""
    text_proj = model.text_projection.T
    V = text_proj / torch.norm(text_proj, p=2, dim=1, keepdim=True)
    cosine_sim = torch.matmul(V, V.T)
    
    # Remove diagonal
    mask = ~torch.eye(512, dtype=bool)
    cosine_values = cosine_sim[mask].flatten().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.hist(cosine_values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Neuron Similarity Distribution (Cs={Cs_value})', fontsize=14)
    plt.axvline(x=0, color='r', linestyle='--', label='Orthogonal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'cosine_hist_Cs{int(Cs_value*100)}.png', dpi=300)
```

---

## Expected Results

### Feature Purity Degradation

| Cs | Avg Abs Cos Sim | Change from Baseline | Interpretation |
|----|-----------------|----------------------|----------------|
| 0.0 | 0.121 | — | **Baseline**: Original alignment |
| 0.1 | 0.143 | +18.2% | Slight loss of orthogonality |
| 0.3 | 0.172 | +42.1% | Moderate feature mixing |
| 0.5 | 0.208 | +71.9% | Significant interference |
| 0.8 | 0.259 | +114.0% | Severe feature corruption |

### Downstream Accuracy Impact

| Cs | CIFAR-10 | CIFAR-100 | Food-101 | Avg Drop |
|----|----------|-----------|----------|----------|
| 0.0 | 45.8% | 22.3% | 38.5% | — |
| 0.1 | 44.2% | 21.1% | 37.1% | -1.8% |
| 0.3 | 41.5% | 18.9% | 34.2% | -4.9% |
| 0.5 | 37.8% | 16.2% | 30.5% | -8.7% |
| 0.8 | 32.1% | 12.8% | 25.3% | -13.8% |

### Key Findings

1. **Strong Correlation**: As Cs increases, purity decreases and accuracy drops
2. **Non-linear Effect**: Impact accelerates at higher misalignment levels
3. **Universal Degradation**: All downstream tasks affected similarly
4. **Mechanistic Link**: Feature purity mediates misalignment's effect on performance

---

## Complete Workflow

### 1. Setup

```bash
# Prepare directory structure
mkdir -p data/captions experiments/3_misalignment/results

# Download pretrained model
wget https://dropbox.com/s/k2e1tgsfmo0afme/cc3m_laclip.pt -O checkpoints/cc3m_laclip.pt
```

### 2. Generate Misaligned Data

```bash
cd experiments/3_misalignment
python generate_misaligned_text.py
```

### 3. Create Training CSVs

```python
# create_csvs.py
import pandas as pd

original_df = pd.read_csv("cc3m_original.csv", header=None)
image_paths = original_df[0].tolist()

for Cs in [0.1, 0.3, 0.5, 0.8]:
    caption_file = f"cc3m_human_10w_Cs{int(Cs*100)}_similarity_simple.txt"
    with open(caption_file) as f:
        captions = [line.strip() for line in f]
    
    df = pd.DataFrame({0: image_paths, 1: captions})
    df.to_csv(f"cc3m_Cs{int(Cs*100)}.csv", header=False, index=False)
```

### 4. Fine-tune Models

```bash
# Train for each Cs value
for cs in 10 30 50 80; do
    torchrun --nproc_per_node=4 train.py \
        --train-data cc3m_Cs${cs}.csv \
        --root /path/to/cc3m \
        --imagenet-root /path/to/imagenet \
        --output-dir results_Cs${cs} \
        --model CLIP_VITB16 \
        --lr 1e-3 --wd 0.5 \
        --batch-size 256 --epochs 10
done
```

### 5. Analyze Results

```python
# analyze_all.py
import torch
import pandas as pd
from collections import OrderedDict
from src.models import CLIP_VITB16

def analyze_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    
    model = CLIP_VITB16(rand_embed=False)
    model.load_state_dict(state_dict)
    
    text_proj = model.text_projection.T
    V = text_proj / torch.norm(text_proj, p=2, dim=1, keepdim=True)
    cos_sim = torch.matmul(V, V.T)
    cos_sim -= torch.diag(torch.diag(cos_sim))
    
    return torch.mean(torch.abs(cos_sim)).item()

results = []
for cs in [0, 10, 30, 50, 80]:
    if cs == 0:
        path = "checkpoints/cc3m_laclip.pt"
    else:
        path = f"results_Cs{cs}/checkpoint.pt"
    
    purity = analyze_checkpoint(path)
    results.append({'Cs': cs/100, 'Purity': purity})
    print(f"Cs={cs/100:.1f}: Purity={purity:.4f}")

df = pd.DataFrame(results)
df.to_csv("final_results.csv", index=False)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(df['Cs'], df['Purity'], 'b-o', linewidth=2, markersize=10)
plt.xlabel('Misalignment Probability', fontsize=14)
plt.ylabel('Avg Abs Cosine Similarity', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('final_purity_analysis.png', dpi=300)
```

---

## Troubleshooting

### Common Issues

1. **Training Instability**:
   ```bash
   --lr 5e-4  # Lower learning rate
   --warmup-epochs 2  # Longer warmup
   ```

2. **Slow Convergence**:
   ```bash
   --epochs 20  # Train longer
   --batch-size 512  # Larger batch
   ```

3. **Memory Issues**:
   ```bash
   --batch-size 128  # Reduce batch size
   --nproc_per_node=2  # Use fewer GPUs
   ```

## Next Steps

1. **Analyze Results**: Generate all plots and statistics
2. **Compare with Random**: Test random shuffling vs similarity-based
3. **Layer-wise Analysis**: Check purity at different network depths
4. **Scaling Study**: Test with larger models (ViT-L/14)

## References

- [Data Quality in Vision-Language Models](https://arxiv.org/abs/2305.20088)
- [TF-IDF for Text Similarity](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
