import torch
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import os
import sys
from collections import OrderedDict
import open_clip
import clip
from sklearn.metrics import silhouette_score

# 添加LaCLIP仓库路径
sys.path.append('/home/sunj11/Documents/LaCLIP')
from models import CLIP_VITB16
from tokenizer import SimpleTokenizer

# 配置
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
ckpt_path = "/home/sunj11/Documents/LaCLIP/result_cc3m_laclip/cc3m_laclip.pt"
save_dir_base = "/home/sunj11/Documents/LaCLIP/result_cc3m_laclip/"

DATASETS = ["CIFAR10", "CIFAR100", "FOOD101", "CALTECH101", "PETS", "DTD", "EUROSAT"]

# 获取模型
def get_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model

def load_standard_model(ckpt_path):
    print(f"加载标准模型 '{ckpt_path}'")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    model = CLIP_VITB16(rand_embed=False)
    model.to(device)
    model.load_state_dict(state_dict, strict=True)
    torch.backends.cudnn.benchmark = True
    return model

# 数据预处理
standard_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    lambda x: x.convert('RGB'),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def load_dataset(dataset_name):
    if dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=standard_transform)
        cls_names = dataset.classes
    elif dataset_name == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=standard_transform)
        cls_names = dataset.classes
    elif dataset_name == "FOOD101":
        dataset = torchvision.datasets.Food101(root='./data', split="test", download=True, transform=standard_transform)
        cls_names = dataset.classes
    elif dataset_name == "CALTECH101":
        dataset = torchvision.datasets.Caltech101(root='./data', download=True, transform=standard_transform, target_type="category")
        cls_names = dataset.categories
    elif dataset_name == "PETS":
        dataset = torchvision.datasets.OxfordIIITPet(root='./data', split="test", download=True, transform=standard_transform, target_types="category")
        cls_names = dataset.classes
    elif dataset_name == "DTD":
        dataset = torchvision.datasets.DTD(root='./data', split="test", download=True, transform=standard_transform)
        cls_names = dataset.classes
    elif dataset_name == "EUROSAT":
        dataset = torchvision.datasets.EuroSAT(root='./data', download=True, transform=standard_transform)
        cls_names = dataset.classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset, cls_names

# 归一化函数
def normalize_tensor(V):
    norm = torch.norm(V, p=2, dim=1, keepdim=True)
    return V / norm

def find_min_off_diagonal_sum(image_proj, text_proj, sam=450):
    image_VVT = torch.matmul(image_proj, image_proj.T)
    text_VVT = torch.matmul(text_proj, text_proj.T)
    image_VVT -= torch.diag(torch.diag(image_VVT))
    text_VVT -= torch.diag(torch.diag(text_VVT))
    total_row_sum = torch.sum(torch.abs(image_VVT), dim=1) + torch.sum(torch.abs(text_VVT), dim=1)
    _, min_row_indices = torch.topk(total_row_sum, sam, largest=False)
    return min_row_indices.tolist()

def find_max_off_diagonal_sum(image_proj, text_proj, sam=450):
    image_VVT = torch.matmul(image_proj, image_proj.T)
    text_VVT = torch.matmul(text_proj, text_proj.T)
    image_VVT -= torch.diag(torch.diag(image_VVT))
    text_VVT -= torch.diag(torch.diag(text_VVT))
    total_row_sum = torch.sum(torch.abs(image_VVT), dim=1) + torch.sum(torch.abs(text_VVT), dim=1)
    _, max_row_indices = torch.topk(total_row_sum, sam, largest=True)
    return max_row_indices.tolist()

def evaluate_zeroshot_classification(model, dataset, cls_names, is_laion_model, batch_size=512, sam=450):
    model.eval()
    text_prompts = [f"a photo of a {cls}" for cls in cls_names]
    if is_laion_model:
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        text_inputs = tokenizer(text_prompts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
    else:
        text_inputs = clip.tokenize(text_prompts).to(device)
        with torch.no_grad():
            text_features = get_model(model).encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    np.random.seed(42)
    # indices = np.random.permutation(len(dataset))[:max_samples]
    indices = np.arange(len(dataset))
    
    image_features_full = []
    labels = []

    for i in tqdm(range(0, len(indices), batch_size)):
        batch_indices = indices[i:i+batch_size]
        batch_images = torch.stack([dataset[idx][0] for idx in batch_indices]).to(device)
        batch_labels = [dataset[idx][1] for idx in batch_indices]

        with torch.no_grad():
            if is_laion_model:
                batch_features = model.encode_image(batch_images)
            else:
                batch_features = get_model(model).encode_image(batch_images)

        image_features_full.append(batch_features.cpu().numpy())
        labels.extend(batch_labels)

    image_features_full = np.concatenate(image_features_full)
    image_features_full /= np.linalg.norm(image_features_full, axis=1, keepdims=True)

    model_for_proj = load_standard_model(ckpt_path)
    image_proj = normalize_tensor(get_model(model_for_proj).image_projection.T)
    text_proj = normalize_tensor(get_model(model_for_proj).text_projection.T)

    min_indices = find_min_off_diagonal_sum(image_proj, text_proj, sam=sam)
    max_indices = find_max_off_diagonal_sum(image_proj, text_proj, sam=sam)

    image_features_min_sparse = np.zeros_like(image_features_full)
    image_features_min_sparse[:, min_indices] = image_features_full[:, min_indices]

    image_features_max_sparse = np.zeros_like(image_features_full)
    image_features_max_sparse[:, max_indices] = image_features_full[:, max_indices]

    results = {}
    for name, features in [
        ('Full Features', image_features_full),
        ('Min Sparse Features', image_features_min_sparse),
        ('Max Sparse Features', image_features_max_sparse)
    ]:
        features_tensor = torch.from_numpy(features).to(device)
        predicted = []
        for i in range(0, len(features_tensor), batch_size):
            batch_features = features_tensor[i:i+batch_size]
            batch_logits = 100.0 * batch_features @ text_features.t()
            batch_predicted = torch.argmax(batch_logits, dim=1).cpu().numpy()
            predicted.extend(batch_predicted)

        correct = np.sum(np.array(predicted) == np.array(labels))
        total = len(labels)
        accuracy = correct / total

        silhouette = silhouette_score(features, labels, metric='cosine')

        results[name] = {
            'Accuracy': accuracy,
            'Silhouette Score': silhouette
        }
    return results

def main():
    checkpoints = {
        "CC3M LaCLIP": ckpt_path,
    }

    sam_values = list(range(200, 501, 50))

    for DATASET in DATASETS:
        dataset, cls_names = load_dataset(DATASET)

        acc_rows = []
        ss_rows = []

        for sam in sam_values:
            print(f"\n===== 当前 dataset={DATASET}, sam={sam} =====")
            for model_name, checkpoint_path in checkpoints.items():
                print(f"\n--- 评估模型: {model_name} on {DATASET} ---")

                try:
                    model = load_standard_model(checkpoint_path)
                    accuracy_results = evaluate_zeroshot_classification(
                        model, dataset, cls_names, is_laion_model=False, sam=sam
                    )

                    acc_rows.append({
                        'sam': sam,
                        'Model': model_name,
                        'Full Features': accuracy_results['Full Features']['Accuracy'],
                        'Min Sparse Features': accuracy_results['Min Sparse Features']['Accuracy'],
                        'Max Sparse Features': accuracy_results['Max Sparse Features']['Accuracy']
                    })

                    ss_rows.append({
                        'sam': sam,
                        'Model': model_name,
                        'Full Features': accuracy_results['Full Features']['Silhouette Score'],
                        'Min Sparse Features': accuracy_results['Min Sparse Features']['Silhouette Score'],
                        'Max Sparse Features': accuracy_results['Max Sparse Features']['Silhouette Score']
                    })

                except Exception as e:
                    print(f"评估模型 {model_name} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()

        acc_df = pd.DataFrame(acc_rows)
        ss_df = pd.DataFrame(ss_rows)

        os.makedirs(save_dir_base, exist_ok=True)
        acc_df.to_csv(os.path.join(save_dir_base, f'{DATASET.lower()}_laclip_accuracy_vs_sam.csv'), index=False)
        ss_df.to_csv(os.path.join(save_dir_base, f'{DATASET.lower()}_laclip_silhouette_vs_sam.csv'), index=False)
        print(f"==> 完成 {DATASET}，结果保存到 {save_dir_base}")

    print("\n所有数据集评估完成！")
    return

if __name__ == "__main__":
    main()