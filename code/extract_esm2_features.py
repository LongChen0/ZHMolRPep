import os
import torch
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm

# ================= 配置路径 =================
# 输入：你上一步生成的 1对1 FASTA 文件夹
INPUT_FASTA_DIR = "/media/zhaolab9/Zhaolab/chenlong/new_RNA_peptide/fasta"
# 输出：专门存放多肽特征的文件夹
OUTPUT_DIR = "/media/zhaolab9/Zhaolab/chenlong/new_RNA_peptide/RNA_Peptide_Features/Peptide_ESM2"
# 本地模型路径 (使用你之前配置的路径)
ESM2_PATH = "/home/zhaolab9/long/llm/esm2_t36_3B_UR50D"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_peptide_seq_from_fasta(filepath):
    """从 Pairwise FASTA 文件中专门提取多肽序列"""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('>')
        for line in lines:
            if not line: continue
            parts = line.strip().split('\n')
            if parts[0].endswith("_Peptide"):
                return "".join(parts[1:])
    return ""

def main():
    print("=" * 60)
    print("🚀 多肽特征提取 (基于 ESM-2 3B)")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 1. 加载模型 (使用 float16 降低显存占用)
    print("正在加载 ESM2 3B 模型...")
    tokenizer = EsmTokenizer.from_pretrained(ESM2_PATH)
    model = EsmModel.from_pretrained(ESM2_PATH, torch_dtype=torch.float16).cuda().eval()
    print("✅ 模型加载完毕！")

    # 2. 扫描 FASTA 文件
    fasta_files = [f for f in os.listdir(INPUT_FASTA_DIR) if f.endswith('.fasta')]
    print(f"共发现 {len(fasta_files)} 个成对序列文件。")

    success_count, skip_count = 0, 0

    # 3. 逐个提取
    for filename in tqdm(fasta_files, desc="提取多肽 ESM2 特征"):
        pair_id = filename.replace(".fasta", "")
        in_path = os.path.join(INPUT_FASTA_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, f"{pair_id}.pt")

        # 支持断点续传
        if os.path.exists(out_path):
            skip_count += 1
            continue

        seq_pep = get_peptide_seq_from_fasta(in_path)
        if not seq_pep:
            continue

        # 编码与推断
        inputs = tokenizer(seq_pep, return_tensors="pt", add_special_tokens=True).to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state

        # 去除头尾特殊字符，并转回 float32 方便后续与其他模块计算
        feat_pep = hidden_states[0, 1:-1, :].cpu().to(torch.float32)

        # 保存为独立的字典文件
        torch.save({
            'pair_id': pair_id,
            'seq_peptide': seq_pep,
            'feat_peptide': feat_pep
        }, out_path)
        
        success_count += 1

    print("\n" + "=" * 50)
    print(f"🎉 肽段特征提取完成！")
    print(f"成功提取: {success_count} 个")
    print(f"跳过(已存在): {skip_count} 个")
    print("=" * 50)

if __name__ == "__main__":
    main()
