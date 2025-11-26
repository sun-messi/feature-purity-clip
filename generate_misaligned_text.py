# %%
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import re

# === 参数 ===
input_txt = "cc3m_human_10w.txt"
Cs_values = [0.1, 0.3, 0.5, 0.8]  # 不同的交换比例
top_k = 10  # 在相似度最高的前 k 个候选中随机选择
num_gpus = 8  # 使用的GPU数量
batch_size = 1000  # 每个批次处理的文本数量

class MultiGPUSimilarityCalculator:
    def __init__(self, num_gpus=8):
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.devices = [f'cuda:{i}' for i in range(self.num_gpus)]
        print(f"🚀 使用 {self.num_gpus} 个GPU: {self.devices}")
        
    def compute_similarity_batch_gpu(self, tfidf_matrix, start_idx, end_idx, device_id):
        """在指定GPU上计算相似度矩阵的一个批次"""
        device = torch.device(f'cuda:{device_id}')
        
        try:
            # 将数据移到GPU
            matrix_tensor = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32).to(device)
            batch_tensor = matrix_tensor[start_idx:end_idx]
            
            # 计算余弦相似度
            matrix_norm = F.normalize(matrix_tensor, p=2, dim=1)
            batch_norm = F.normalize(batch_tensor, p=2, dim=1)
            
            # 计算相似度
            similarity_batch = torch.mm(batch_norm, matrix_norm.t())
            
            # 移回CPU
            result = similarity_batch.cpu().numpy()
            
            # 清理GPU内存
            del matrix_tensor, batch_tensor, matrix_norm, batch_norm, similarity_batch
            torch.cuda.empty_cache()
            
            return start_idx, end_idx, result
            
        except Exception as e:
            print(f"❌ GPU {device_id} 计算批次 [{start_idx}:{end_idx}] 时出错: {e}")
            return start_idx, end_idx, None

def preprocess_text(text):
    """简单的文本预处理"""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_texts_parallel(texts, num_workers=None):
    """并行预处理文本"""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)
    
    print(f"🔄 使用 {num_workers} 个进程并行预处理文本...")
    
    with mp.Pool(num_workers) as pool:
        processed_texts = list(tqdm(
            pool.imap(preprocess_text, texts, chunksize=1000),
            total=len(texts),
            desc="预处理文本"
        ))
    
    return processed_texts

def compute_similarity_matrix_multigpu(texts, calculator, batch_size=1000):
    """使用多GPU计算文本间的余弦相似度矩阵"""
    print("🔄 正在并行预处理文本...")
    processed_texts = preprocess_texts_parallel(texts)
    
    print("🔄 正在计算 TF-IDF 向量...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    print(f"📊 TF-IDF 矩阵形状: {tfidf_matrix.shape}")
    
    # 计算批次
    total_texts = len(texts)
    num_batches = (total_texts + batch_size - 1) // batch_size
    
    print(f"🚀 开始多GPU相似度计算，共 {num_batches} 个批次...")
    
    # 初始化相似度矩阵
    similarity_matrix = np.zeros((total_texts, total_texts), dtype=np.float32)
    
    # 创建批次任务
    batch_tasks = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_texts)
        device_id = i % calculator.num_gpus
        batch_tasks.append((start_idx, end_idx, device_id))
    
    # 使用线程池管理GPU任务
    with ThreadPoolExecutor(max_workers=calculator.num_gpus) as executor:
        # 提交任务
        future_to_batch = {
            executor.submit(
                calculator.compute_similarity_batch_gpu,
                tfidf_matrix, start_idx, end_idx, device_id
            ): (start_idx, end_idx, device_id)
            for start_idx, end_idx, device_id in batch_tasks
        }
        
        # 收集结果
        completed = 0
        for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="GPU计算进度"):
            start_idx, end_idx, result = future.result()
            
            if result is not None:
                similarity_matrix[start_idx:end_idx] = result
                completed += 1
            else:
                print(f"⚠️ 批次 [{start_idx}:{end_idx}] 计算失败")
    
    print(f"✅ 完成 {completed}/{num_batches} 个批次的计算")
    return similarity_matrix

def find_similar_candidates(similarity_matrix, idx, top_k, exclude_indices=None):
    """找到与指定索引最相似的前 top_k 个候选"""
    similarities = similarity_matrix[idx].copy()
    similarities[idx] = -1  # 排除自身
    
    if exclude_indices:
        for ex_idx in exclude_indices:
            similarities[ex_idx] = -1
    
    # 找到相似度最高的 top_k 个索引
    top_indices = np.argsort(similarities)[-top_k:]
    # 过滤掉相似度为负的（即被排除的）
    top_indices = [i for i in top_indices if similarities[i] >= 0]
    
    return top_indices

def simple_similarity_swap(lines, similarity_matrix, Cs):
    """简单的基于相似度交换：交换到刚好超过目标比例就停止"""
    total_lines = len(lines)
    target_swap_count = int(Cs * total_lines)
    
    print(f"🎯 目标交换行数: {target_swap_count} (目标比例: {Cs:.1%})")
    
    final_lines = lines.copy()
    used_indices = set()
    swap_pairs = []
    swapped_lines = 0
    
    # 随机排列所有索引
    all_indices = np.arange(total_lines)
    np.random.shuffle(all_indices)
    
    for source_idx in tqdm(all_indices, desc="寻找交换对"):
        if swapped_lines >= target_swap_count:
            print(f"✅ 已达到目标交换数量，停止交换")
            break
            
        if source_idx in used_indices:
            continue
        
        # 找到与当前行最相似的候选
        candidates = find_similar_candidates(
            similarity_matrix, 
            source_idx, 
            top_k, 
            exclude_indices=used_indices
        )
        
        if candidates:
            # 从候选中随机选择一个进行交换
            target_idx = np.random.choice(candidates)
            
            # 执行交换
            final_lines[source_idx], final_lines[target_idx] = final_lines[target_idx], final_lines[source_idx]
            
            # 记录已使用的索引和交换对
            used_indices.add(source_idx)
            used_indices.add(target_idx)
            swap_pairs.append((source_idx, target_idx))
            swapped_lines += 2  # 每次交换影响2行
    
    actual_swap_rate = swapped_lines / total_lines
    print(f"✅ 完成交换: {len(swap_pairs)}对 = {swapped_lines}行")
    print(f"📊 实际交换率: {actual_swap_rate:.1%} (目标: {Cs:.1%})")
    
    return final_lines, swap_pairs

# === 主程序 ===
def main():
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，无法使用GPU加速")
        return
    
    print(f"🔥 检测到 {torch.cuda.device_count()} 个GPU")
    
    # 读取原始文本
    print("📖 读取原始文本...")
    with open(input_txt, "r", encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"📊 总行数: {total_lines}")
    
    # 初始化多GPU计算器
    calculator = MultiGPUSimilarityCalculator(num_gpus=num_gpus)
    
    # 计算相似度矩阵（只计算一次）
    texts = [line.strip() for line in lines]
    similarity_matrix = compute_similarity_matrix_multigpu(
        texts, calculator, batch_size=batch_size
    )
    print("✅ 相似度矩阵计算完成")
    
    # 批量处理不同的Cs值
    for Cs in Cs_values:
        print(f"\n🔀 处理 Cs = {Cs} ...")
        
        # 设置随机种子
        np.random.seed(42)
        
        output_txt = f"cc3m_human_10w_Cs{int(Cs * 100)}_similarity_simple.txt"
        
        # === 简单交换：交换到刚好超过目标比例就停止 ===
        final_lines, swap_pairs = simple_similarity_swap(lines, similarity_matrix, Cs)
        
        # === 写入输出文件 ===
        print("💾 保存结果...")
        with open(output_txt, "w", encoding='utf-8') as f:
            f.writelines(final_lines)
        
        print(f"✅ 保存成功: {output_txt}")
        
        # === 显示交换示例 ===
        if swap_pairs:
            print(f"\n📝 交换示例 (Cs={Cs}):")
            for i, (idx1, idx2) in enumerate(swap_pairs[:2]):
                sim_score = similarity_matrix[idx1, idx2]
                print(f"交换对 {i+1} (相似度: {sim_score:.3f}):")
                print(f"  位置 {idx1}: {lines[idx2].strip()[:60]}...")
                print(f"  位置 {idx2}: {lines[idx1].strip()[:60]}...")
    
    print(f"\n🎉 所有文件生成完成！")
    print("生成的文件:")
    for Cs in Cs_values:
        print(f"  - cc3m_human_10w_Cs{int(Cs * 100)}_similarity_simple.txt")
    
    # 清理GPU内存
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

# %%


# %%


# %%
import torch
import numpy as np
from collections import OrderedDict
import os
from models import CLIP_VITB16

# 加载模型
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    model = CLIP_VITB16(rand_embed=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

# 向量归一化
def normalize_tensor(V):
    norm = torch.norm(V, p=2, dim=1, keepdim=True)
    return V / norm

# 计算平均绝对Cosine相似度
def compute_avg_abs_cos_sim(V):
    V = normalize_tensor(V)
    cosine_sim = torch.matmul(V, V.T)  # (512, 512)
    cosine_sim = cosine_sim - torch.diag(torch.diag(cosine_sim))  # 去掉对角线
    avg_abs_cos = torch.mean(torch.abs(cosine_sim))
    return avg_abs_cos.item()

# 主程序
def main():
    checkpoints = {
        "C_s=0": "finetune_result_CLIP/checkpoint.pt",
        "C_s=0.1": "finetune_result_Cs10_similarity/checkpoint.pt",
        "C_s=0.3": "finetune_result_Cs30_similarity/checkpoint.pt",
        "C_s=0.5": "finetune_result_Cs50_similarity/checkpoint.pt",
        "C_s=0.8": "finetune_result_Cs80_similarity/checkpoint.pt",
    }

    Cs_values = []
    avg_abs_cos_sims = []

    for label, ckpt_path in checkpoints.items():
        print(f"处理 {label} ...")
        model = load_model(ckpt_path)
        image_proj = model.text_projection  # (768,512)
        image_proj = image_proj.T  # (512,768)

        avg_abs_cos_sim = compute_avg_abs_cos_sim(image_proj)

        # 记录
        Cs = float(label.split('=')[1])
        Cs_values.append(Cs)
        avg_abs_cos_sims.append(avg_abs_cos_sim)

        print(f"{label}: 平均绝对CosSim = {avg_abs_cos_sim:.4f}")

    # 保存结果
    os.makedirs("image", exist_ok=True)
    np.savez("image/avg_abs_cos_sim_vs_cs_similarity.npz", Cs=Cs_values, avg_abs_cos_sims=avg_abs_cos_sims)

    # 画图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.plot(Cs_values, avg_abs_cos_sims, 'b-o', linewidth=10, markersize=25)
    plt.xlabel(r'Shuffling Probability $C_m$', labelpad=15)
    plt.ylabel('Average Absolute Cosine Similarity', labelpad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("image/avg_abs_cos_sim_vs_cs_similarity.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ 计算并保存完成！")
    print("📁 保存的文件:")
    print("  - image/avg_abs_cos_sim_vs_cs_similarity.npz")
    print("  - image/avg_abs_cos_sim_vs_cs_similarity.png")

if __name__ == "__main__":
    main()

# %%
import numpy as np
import pandas as pd

def load_and_compare_npz():
    # 读取两个npz文件
    random_data = np.load("image/avg_abs_cos_sim_vs_cs.npz")
    similarity_data = np.load("image/avg_abs_cos_sim_vs_cs_similarity.npz")
    
    # 提取数据
    random_cs = random_data['Cs']
    random_cos_sim = random_data['avg_abs_cos_sims']
    
    similarity_cs = similarity_data['Cs']
    similarity_cos_sim = similarity_data['avg_abs_cos_sims']
    
    # 创建对比表格
    comparison_df = pd.DataFrame({
        'Cs_Value': random_cs,
        'Random_Shuffle': random_cos_sim,
        'Similarity_Shuffle': similarity_cos_sim,
        'Difference': similarity_cos_sim - random_cos_sim,
        'Relative_Change(%)': ((similarity_cos_sim - random_cos_sim) / random_cos_sim) * 100
    })
    
    # 格式化显示
    print("=" * 80)
    print("📊 Random Shuffle vs Similarity-based Shuffle 对比表")
    print("=" * 80)
    print(f"{'Cs值':<8} {'随机交换':<12} {'相似度交换':<12} {'差值':<12} {'相对变化(%)':<12}")
    print("-" * 80)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['Cs_Value']:<8.1f} {row['Random_Shuffle']:<12.4f} {row['Similarity_Shuffle']:<12.4f} "
              f"{row['Difference']:<12.4f} {row['Relative_Change(%)']:<12.2f}")
    
    print("=" * 80)
    
    # 保存为CSV文件
    comparison_df.to_csv("image/shuffle_comparison_table.csv", index=False, float_format='%.4f')
    print("📁 表格已保存为: image/shuffle_comparison_table.csv")
    
    return comparison_df

if __name__ == "__main__":
    df = load_and_compare_npz()

# %%



