import os
import pandas as pd
import requests
import time
import torch
from tqdm import tqdm

# ================= 配置 =================
CSV_FILE = 'dataset_split_culled.csv'
OUT_DIR = 'monomer_pdbs/peptides'
ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"

def fold_peptides():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    df = pd.read_csv(CSV_FILE)
    print(f"📦 共找到 {len(df)} 个样本，准备开始 ESMFold 结构预测...")
    
    # ================= 稳健地获取序列 =================
    seqs = []
    for idx, row in df.iterrows():
        # 1. 尝试从常见的列名读取
        if 'peptide_sequence' in df.columns:
            seqs.append(row['peptide_sequence'])
        elif 'Protein1' in df.columns:
            seqs.append(row['Protein1'])
        elif 'protein_sequence' in df.columns:
            seqs.append(row['protein_sequence'])
        else:
            # 2. 如果 CSV 找不到序列，直接打开对应的 .pt 文件强行读取！
            pt_path = row.get('pt_file_path', '')
            if os.path.exists(pt_path):
                data = torch.load(pt_path, map_location='cpu', weights_only=False)
                # 尝试不同的可能键名
                seq = data.get('seq_peptide', data.get('peptide_seq', data.get('seq_prot', '')))
                if not seq:
                    print(f"\n⚠️ 警告: 无法在 {pt_path} 中找到序列，跳过此样本。")
                    seq = "UNKNOWN"
                seqs.append(seq)
            else:
                print(f"\n⚠️ 警告: 找不到 .pt 文件 {pt_path}")
                seqs.append("UNKNOWN")
                
    # 兼容不同的 id 列名
    sample_ids = df['sample_id'].tolist() if 'sample_id' in df.columns else df['id'].tolist()
    
    # ================= 开始 API 预测 =================
    success_count = 0
    for i in tqdm(range(len(df)), desc="ESMFold API 预测中"):
        sid = sample_ids[i]
        seq = seqs[i].upper().strip()
        
        if seq == "UNKNOWN":
            continue
            
        pdb_path = os.path.join(OUT_DIR, f"{sid}_peptide.pdb")
        
        # 如果已经预测过了，就跳过（支持断点续传）
        if os.path.exists(pdb_path):
            success_count += 1
            continue
            
        try:
            # 调用 ESMFold API
            response = requests.post(ESMFOLD_API, data=seq, timeout=60)
            if response.status_code == 200:
                with open(pdb_path, 'w') as f:
                    f.write(response.text)
                success_count += 1
            else:
                print(f"\n⚠️ 样本 {sid} 预测失败，状态码: {response.status_code}")
            
            # 礼貌性延迟，防止被 API 封禁 IP
            time.sleep(1.0)
            
        except Exception as e:
            print(f"\n❌ 请求样本 {sid} 时发生错误: {e}")
            time.sleep(5.0) # 出错后多等一会儿
            
    print(f"\n🎉 预测完成！成功生成 {success_count}/{len(df)} 个多肽的 PDB 文件。")
    print(f"文件保存在: {OUT_DIR}/")

if __name__ == "__main__":
    fold_peptides()
