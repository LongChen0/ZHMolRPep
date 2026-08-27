import os
import gc
import torch
from tqdm import tqdm
from multimolecule import RnaTokenizer, RiNALMoModel

# ================= 配置路径 =================
INPUT_FASTA_DIR = "/media/zhaolab9/Zhaolab/chenlong/new_RNA_peptide/fasta"
OUTPUT_DIR = "/media/zhaolab9/Zhaolab/chenlong/new_RNA_peptide/RNA_Peptide_Features/RNA_RiNALMo"
LOCAL_MODEL_PATH = "/home/zhaolab9/long/llm/rinalmo-giga"

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_rna_seq_from_fasta(filepath):
    """从 Pairwise FASTA 文件中专门提取 RNA 序列"""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('>')
        for line in lines:
            if not line: continue
            parts = line.strip().split('\n')
            if parts[0].endswith("_RNA"):
                return "".join(parts[1:]).upper()
    return ""

def save_atomically(data_dict, target_path):
    """原子保存：防止产生 0kb 损坏文件"""
    tmp_path = target_path + ".tmp"
    torch.save(data_dict, tmp_path)
    os.replace(tmp_path, target_path) # 原子级重命名覆盖

def extract_features(seq_rna, model, tokenizer, device):
    """执行模型前向传播的独立函数，便于局部变量及时销毁"""
    raw_len = len(seq_rna)
    encoded_input = tokenizer(seq_rna, return_tensors="pt")
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    valid_len = input_ids.shape[1]

    # 使用 inference_mode 比 no_grad 更省显存
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
    
    # 立即转移到 CPU 并转为 float32
    seq_emb = last_hidden_state[0].cpu().to(torch.float32)

    # 裁剪特殊 Token
    if valid_len == raw_len + 2:
        seq_emb = seq_emb[1:-1, :]
    else:
        seq_emb = seq_emb[1 : 1 + raw_len, :]

    # 显式删除计算图变量，释放显存
    del input_ids, attention_mask, output, last_hidden_state
    return seq_emb

def main():
    print("=" * 60)
    print("🚀 RNA 特征提取 (基于 RiNALMo-Giga | 工业级强壮版)")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # ================= 阶段 1：GPU 高速提取 =================
    print("\n[阶段 1] 正在加载 RiNALMo 模型到 GPU (float16)...")
    tokenizer = RnaTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model = RiNALMoModel.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.float16, local_files_only=True)
    model.to(DEVICE).eval()
    print("✅ GPU 模型加载成功！")

    fasta_files = [f for f in os.listdir(INPUT_FASTA_DIR) if f.endswith('.fasta')]
    success_count, skip_count = 0, 0
    failed_pairs = [] # 记录 GPU 处理失败的名单

    for filename in tqdm(fasta_files, desc="GPU 提取中"):
        pair_id = filename.replace(".fasta", "")
        in_path = os.path.join(INPUT_FASTA_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, f"{pair_id}.pt")

        # 断点续传：检查文件是否存在且大小大于 0
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skip_count += 1
            continue

        seq_rna = get_rna_seq_from_fasta(in_path)
        if not seq_rna:
            continue

        try:
            seq_emb = extract_features(seq_rna, model, tokenizer, DEVICE)
            save_atomically({
                'pair_id': pair_id,
                'seq_rna': seq_rna,
                'feat_rna': seq_emb
            }, out_path)
            success_count += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                failed_pairs.append((pair_id, seq_rna))
            else:
                print(f"\n[!] 处理 {pair_id} 发生未知错误: {e}")
                failed_pairs.append((pair_id, seq_rna))
        finally:
            # 无论成功还是失败，强制执行深度垃圾回收
            gc.collect()
            torch.cuda.empty_cache()

    # ================= 阶段 2：CPU 自动拯救 =================
    if failed_pairs:
        print("\n" + "=" * 50)
        print(f"🚑 触发自动拯救机制！有 {len(failed_pairs)} 个长序列导致 GPU 溢出。")
        print("正在将模型转移至 CPU 并转换为 float32 以保证兼容性 (这需要一点时间)...")
        
        # 卸载 GPU 显存
        model = model.float().to('cpu') 
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ 转移完成，开始利用大内存进行慢速提取...")

        cpu_success = 0
        for pair_id, seq_rna in tqdm(failed_pairs, desc="CPU 拯救中"):
            out_path = os.path.join(OUTPUT_DIR, f"{pair_id}.pt")
            try:
                seq_emb = extract_features(seq_rna, model, tokenizer, 'cpu')
                save_atomically({
                    'pair_id': pair_id,
                    'seq_rna': seq_rna,
                    'feat_rna': seq_emb
                }, out_path)
                cpu_success += 1
            except Exception as e:
                print(f"\n[!] CPU 处理 {pair_id} 依然失败: {e}")

    # ================= 总结报告 =================
    print("\n" + "=" * 50)
    print(f"🎉 全部特征提取任务彻底完成！")
    print(f"GPU 成功提取: {success_count} 个")
    if failed_pairs:
        print(f"CPU 成功拯救: {cpu_success}/{len(failed_pairs)} 个")
    print(f"跳过(已存在且完好): {skip_count} 个")
    print("=" * 50)

if __name__ == "__main__":
    main()
