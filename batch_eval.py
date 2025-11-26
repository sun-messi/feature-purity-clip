import os
import torch
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import sys
from collections import OrderedDict
import open_clip
import clip
from sklearn.metrics import silhouette_score

sys.path.append('/home/sunj11/Documents/LaCLIP')
from models import CLIP_VITB16

base_ckpt_paths = {
    "Cs0": "/home/sunj11/Documents/LaCLIP/checkpoint/cc3m_laclip.pt",
    # "Cs10": "/home/sunj11/Documents/LaCLIP/finetune_result_Cs10/checkpoint.pt",
    # "Cs30": "/home/sunj11/Documents/LaCLIP/finetune_result_Cs30/checkpoint.pt",
    # "Cs50": "/home/sunj11/Documents/LaCLIP/finetune_result_Cs50/checkpoint.pt",
    # "Cs80": "/home/sunj11/Documents/LaCLIP/finetune_result_Cs80/checkpoint.pt"
}

def get_model(model):
    return model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model

def load_standard_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    model = CLIP_VITB16(rand_embed=False).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model.load_state_dict({f'module.{k}': v for k, v in state_dict.items()}, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    torch.backends.cudnn.benchmark = True
    return model

def evaluate(dataset, cls_names, model, is_laion_model, device, batch_size=1024):
    model.eval()
    text_prompts = [f"a photo of a {cls.replace('_',' ').replace('/',' ')}" for cls in cls_names]
    tokenizer = open_clip.get_tokenizer('ViT-B-16') if is_laion_model else clip.tokenize
    text_inputs = tokenizer(text_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs) if is_laion_model else get_model(model).encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    indices = np.arange(len(dataset))
    image_features_full, labels = [], []
    for i in tqdm(range(0, len(indices), batch_size)):
        batch = indices[i:i+batch_size]
        imgs = torch.stack([dataset[idx][0] for idx in batch]).to(device)
        lbls = [dataset[idx][1] for idx in batch]
        with torch.no_grad():
            feats = model.encode_image(imgs) if is_laion_model else get_model(model).encode_image(imgs)
        image_features_full.append(feats.cpu().numpy())
        labels.extend(lbls)

    image_features_full = np.concatenate(image_features_full)
    image_features_full /= np.linalg.norm(image_features_full, axis=1, keepdims=True)

    features_tensor = torch.from_numpy(image_features_full).to(device)
    predicted = []
    for i in range(0, len(features_tensor), batch_size):
        logits = 100.0 * features_tensor[i:i+batch_size] @ text_features.t()
        predicted.extend(torch.argmax(logits, dim=1).cpu().numpy())

    acc = np.mean(np.array(predicted) == np.array(labels))
    sil = silhouette_score(image_features_full, labels, metric='cosine')
    return {"Accuracy": acc, "Silhouette Score": sil}

def load_dataset(name):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    root = './data'
    if name == "CIFAR10":
        d = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform); return d, d.classes
    elif name == "CIFAR100":
        d = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform); return d, d.classes
    elif name == "FOOD101":
        d = torchvision.datasets.Food101(root=root, split="test", download=True, transform=transform); return d, d.classes
    elif name == "CALTECH101":
        d = torchvision.datasets.Caltech101(root=root, download=True, transform=transform, target_type="category"); return d, d.categories
    elif name == "PETS":
        d = torchvision.datasets.OxfordIIITPet(root=root, split="test", download=True, transform=transform, target_types="category"); return d, d.classes
    elif name == "STL10":
        d = torchvision.datasets.STL10(root=root, split="test", download=True, transform=transform); return d, d.classes
    else:
        raise ValueError(name)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["CALTECH101", "CIFAR10", "CIFAR100", "FOOD101", "PETS", "STL10"]

    for tag, ckpt_path in base_ckpt_paths.items():
        print(f"\n\n======== {tag} ========")
        save_dir = f"/home/sunj11/Documents/LaCLIP/{tag}_full512_results"
        os.makedirs(save_dir, exist_ok=True)
        for dataset_name in datasets:
            print(f"Processing {dataset_name}")
            dataset, cls_names = load_dataset(dataset_name)
            model = load_standard_model(ckpt_path, device)
            result = evaluate(dataset, cls_names, model, is_laion_model=False, device=device)
            df = pd.DataFrame([result])
            df.to_csv(os.path.join(save_dir, f'{dataset_name.lower()}_full_feature_acc_ss.csv'), index=False)
            print(f"{dataset_name}: {result}")

if __name__ == "__main__":
    main()