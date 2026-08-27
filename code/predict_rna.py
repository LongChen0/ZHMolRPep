import os
import pandas as pd
import torch
import subprocess
import shutil
import re
from tqdm import tqdm

# ================= 配置参数 =================
CSV_FILE = 'dataset_split_culled.csv'
FASTA_DIR = os.path.abspath('rna_fasta')          # 第一步：保存 fasta 的目录
OUT_DIR = os.path.abspath('monomer_pdbs/rnas')    # 第二步：保存最终 pdb 的目录

# 👇 你的真实 RhoFold 绝对路径
RHOFOLD_DIR = "/home/zhaolab9/long/rhofold/RhoFold" 
# ============================================

def step1_generate_fasta():
    """第一步：提取、清洗并保存所有序列为 .fasta 文件"""
    os.makedirs(FASTA_DIR, exist_ok=True)
    df = pd.read_csv(CSV_FILE)
    print(f"\n🚀 [STEP 1] 开始从 {CSV_FILE} 提取序列并生成 FASTA 文件...")
    
    seqs = []
    for idx, row in df.iterrows():
        # 尝试从各种可能的列中寻找 RNA 序列
        if 'RNA_sequence' in df.columns:
            seqs.append(row['RNA_sequence'])
        elif 'RNA_aa_code' in df.columns:
            seqs.append(row['RNA_aa_code'])
        else:
            pt_path = row.get('pt_file_path', '')
            if os.path.exists(pt_path):
                data = torch.load(pt_path, map_location='cpu')
                seq = data.get('seq_rna', data.get('rna_seq', ''))
                seqs.append(seq if seq else "UNKNOWN")
            else:
                seqs.append("UNKNOWN")
                
    sample_ids = df['sample_id'].tolist() if 'sample_id' in df.columns else df['id'].tolist()
    
    valid_count = 0
    for i in tqdm(range(len(df)), desc="生成 FASTA"):
        sid = sample_ids[i]
        raw_seq = seqs[i].upper().strip()
        
        if raw_seq == "UNKNOWN":
            continue
            
        # 🌟 核心清洗：把 T 换成 U，把非 AUGC 的异常碱基（比如 I, N）强行替换为 A，防止 RhoFold 崩溃
        seq = raw_seq.replace("T", "U")
        seq = re.sub(r'[^AUGC]', 'A', seq) 
        
        fasta_path = os.path.join(FASTA_DIR, f"{sid}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{sid}\n{seq}\n")
        valid_count += 1
        
    print(f"✅ [STEP 1 完成] 成功生成并清洗了 {valid_count} 个 .fasta 文件，保存在 {FASTA_DIR} 目录下。")
    return valid_count

def step2_predict_from_fasta():
    """第二步：读取 fasta 文件夹，批量调用 RhoFold 进行预测"""
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 获取所有的 fasta 文件
    fasta_files = [f for f in os.listdir(FASTA_DIR) if f.endswith('.fasta')]
    print(f"\n🚀 [STEP 2] 开始读取 {len(fasta_files)} 个 FASTA 文件并执行 RhoFold 预测...")
    
    # 🌟 绝对断网环境变量，防止 Hugging Face 连接超时
    my_env = os.environ.copy()
    my_env["HF_ENDPOINT"] = "https://hf-mirror.com"
    my_env["HF_HUB_OFFLINE"] = "1"
    my_env["TRANSFORMERS_OFFLINE"] = "1"
    
    success_count = 0
    
    for fasta_name in tqdm(fasta_files, desc="RhoFold 极速推断"):
        sid = fasta_name.replace('.fasta', '')
        final_pdb_path = os.path.join(OUT_DIR, f"{sid}_rna.pdb")
        
        # 断点续传，如果已经生成了 PDB 就跳过
        if os.path.exists(final_pdb_path):
            success_count += 1
            continue
            
        fasta_path = os.path.join(FASTA_DIR, fasta_name)
        tmp_out_dir = os.path.abspath(f"tmp_rhofold_out_{sid}")
        os.makedirs(tmp_out_dir, exist_ok=True)
        
        # 构建推断命令（加入 --relax_steps 0 跳过耗时且易报错的物理松弛）
        cmd = [
            "python", "inference.py",
            "--input_fas", fasta_path,
            "--single_seq_pred", "True",           
            "--output_dir", tmp_out_dir,
            "--ckpt", "pretrained/RhoFold_pretrained.pt",
            "--device", "cuda:0",
            "--relax_steps", "0"  
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=RHOFOLD_DIR, env=my_env)
            
            # 由于关掉了 Relax，现在必定只生成 unrelaxed_model.pdb
            unrelaxed_pdb = os.path.join(tmp_out_dir, "unrelaxed_model.pdb")
            
            if os.path.exists(unrelaxed_pdb):
                shutil.copy(unrelaxed_pdb, final_pdb_path)
                success_count += 1
            else:
                if "CUDA out of memory" in result.stderr:
                    print(f"\n⚠️ {sid} 跳过: 序列过长导致显存不足 (OOM)。")
                else:
                    print(f"\n❌ {sid} 预测失败！\n报错信息: {result.stderr[-300:]}")
                
        except Exception as e:
            print(f"\n执行出错: {e}")
            
        finally:
            # 清理 RhoFold 的临时输出目录
            if os.path.exists(tmp_out_dir):
                shutil.rmtree(tmp_out_dir)

    print(f"\n🎉 [STEP 2 完成] 预测结束！成功生成 {success_count}/{len(fasta_files)} 个 RNA 的 PDB 文件。")

if __name__ == "__main__":
    # 先生成全部 Fasta
    step1_generate_fasta()
    
    # 再执行结构预测
    step2_predict_from_fasta()
