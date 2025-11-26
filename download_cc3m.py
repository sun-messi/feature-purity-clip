import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# === 配置 ===
csv_path = "cc3m_original.csv"
output_root = "/home/sunj11/Documents/LaCLIP/cc3m3"
os.makedirs(output_root, exist_ok=True)
max_retries = 1
num_threads = 64

# === 加载 CSV ===
df = pd.read_csv(csv_path, header=None)
paths = df[0].tolist()
urls = df[2].tolist()

assert len(paths) == len(urls), "路径和 URL 数量不匹配"

# === 下载函数 ===
def download_one(args):
    rel_path, url = args
    if pd.isna(url) or not isinstance(url, str) or url.strip() == "":
        return None
    save_path = os.path.join(output_root, rel_path)
    if os.path.exists(save_path):
        return None
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for _ in range(max_retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                return None
        except:
            pass
    return (rel_path, url)

# === 下载并显示实时进度 ===
failures = []
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    with tqdm(total=len(paths), desc="Downloading") as pbar:
        for result in executor.map(download_one, zip(paths, urls)):
            pbar.update(1)
            if result:
                failures.append(result)

# === 写入失败列表 ===
if failures:
    with open("download_failed.txt", "w") as f:
        for rel_path, url in failures:
            f.write(f"{rel_path},{url}\n")
    print(f"⚠️ 有 {len(failures)} 张图像下载失败，详情见 download_failed.txt")
else:
    print("✅ 全部图像下载成功！")