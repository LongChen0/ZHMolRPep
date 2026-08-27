import os
import torch
import numpy as np
from tqdm import tqdm
import pyrosetta
from pyrosetta import rosetta
from dataset import RnaPeptideDataset
from model import ZHMolTopoRPI_Network
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 1. 路径与参数配置 =================
BASE_DIR = "/home/zhaolab/long/new_RNA_peptide"
RNA_DIR = os.path.join(BASE_DIR, "RNA_Peptide_Features/RNA_RiNALMo")
PEP_DIR = os.path.join(BASE_DIR, "RNA_Peptide_Features/Peptide_ESM2")
TOPO_DIR = os.path.join(BASE_DIR, "features_3d/ZHMolTopo_3D_monomer/raw_numpy")
LABEL_DIR = os.path.join(BASE_DIR, "RNA_Peptide_Pairwise/labels")
SPLIT_DIR = os.path.join(BASE_DIR, "Dataset_Splits")
MODEL_WEIGHTS = "best_zhmoltoporpi_model.pth"

MONOMER_PEP_DIR = os.path.join(BASE_DIR, "RNA_Peptide_Pairwise/monomer_pdbs/peptides")
MONOMER_RNA_DIR = os.path.join(BASE_DIR, "RNA_Peptide_Pairwise/monomer_pdbs/rnas")
OUT_COMPLEX_DIR = os.path.join(BASE_DIR, "model/PCA_monomer/2/predict_3D/mypredict_3D")
TMP_DIR = os.path.join(OUT_COMPLEX_DIR, "tmp")

os.makedirs(OUT_COMPLEX_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 2. PyRosetta 初始化函数 =================
def init_pyrosetta_worker():
    """子进程初始化 PyRosetta，每个子进程只执行一次"""
    if not hasattr(pyrosetta, '_initialized'):
        pyrosetta.init('-hb_cen_soft -constant_seed -relax:default_repeats 2 -default_max_cycles 200 -out:level 100 -mute all')
        pyrosetta._initialized = True

# ================= 3. 核心功能函数 =================
def generate_binary_elite_constraints(prob_matrix, rst_file, top_k_ratio=0.5):
    """终极二分类约束生成器：抛弃绝对阈值，引入 Top-K 精英筛选，过滤 84% 噪声"""
    lines = []
    L_r, L_p = prob_matrix.shape
    flat_probs = prob_matrix.flatten()
    L_min = min(L_r, L_p)
    k = max(5, int(L_min * top_k_ratio))
    k = min(k, 20)
    top_k_indices = np.argsort(flat_probs)[::-1][:k]
    
    for idx in top_k_indices:
        r = idx // L_p
        p = idx % L_p
        prob = prob_matrix[r, p]
        if prob < 0.2:
            continue
            
        if prob >= 0.9:
            upper_bound = 4.5
            sd = 0.5
        elif prob >= 0.7:
            upper_bound = 6.0
            sd = 0.8
        else:
            upper_bound = 8.0
            sd = 1.0
            
        weight = min(prob * 2.0, 1.0)
        line = f"AtomPair CA {p + 1}P C4' {r + 1}R SCALARWEIGHTEDFUNC {weight:.3f} BOUNDED 2.0 {upper_bound:.1f} {sd:.1f} 0.5 tag"
        lines.append(line)
        
    with open(rst_file, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)

def combine_and_renumber_pdbs(pep_pdb, rna_pdb, out_pdb):
    def process_lines(pdb_file, target_chain):
        processed = []
        with open(pdb_file, 'r') as f:
            current_res = None
            res_counter = 0
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    res_idx_str = line[22:26].strip()
                    if current_res != res_idx_str:
                        current_res = res_idx_str
                        res_counter += 1
                    new_line = line[:21] + target_chain + line[22:]
                    new_res_str = f"{res_counter:>4}"
                    new_line = new_line[:22] + new_res_str + new_line[26:]
                    processed.append(new_line)
        return processed

    pep_lines = process_lines(pep_pdb, "P")
    rna_lines = process_lines(rna_pdb, "R")
    with open(out_pdb, 'w') as f:
        f.writelines(pep_lines)
        f.write("TER\n")
        f.writelines(rna_lines)
        f.write("END\n")

def run_pyrosetta_docking(start_pdb, rst_file, out_pdb):
    pose = pyrosetta.pose_from_pdb(start_pdb)
    sf_fa = pyrosetta.create_score_function('ref2015')
    sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 20.0)
    
    switch = rosetta.protocols.simple_moves.SwitchResidueTypeSetMover("fa_standard")
    switch.apply(pose)
    
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(rst_file)
    constraints.add_constraints(True)
    constraints.apply(pose)
    
    mmap = pyrosetta.MoveMap()
    mmap.set_bb(False)
    mmap.set_chi(True)
    mmap.set_jump(True)
    
    relax = rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf_fa)
    relax.max_iter(500)
    relax.dualspace(True)
    relax.set_movemap(mmap)
    relax.apply(pose)
    pose.dump_pdb(out_pdb)

def process_single_complex(args):
    """单个复合体处理的完整逻辑，供多进程调用"""
    pair_id, probs, pep_pdb, rna_pdb, tmp_dir, out_dir = args
    try:
        init_pyrosetta_worker() # 确保当前子进程已初始化
        
        rst_file = os.path.join(tmp_dir, f"{pair_id}.rst")
        generate_binary_elite_constraints(probs, rst_file, top_k_ratio=0.5)
        
        init_pdb = os.path.join(tmp_dir, f"{pair_id}_init.pdb")
        combine_and_renumber_pdbs(pep_pdb, rna_pdb, init_pdb)
        
        final_pdb = os.path.join(out_dir, f"{pair_id}_complex.pdb")
        run_pyrosetta_docking(init_pdb, rst_file, final_pdb)
        
        # 清理临时文件（可选）
        if os.path.exists(rst_file): os.remove(rst_file)
        if os.path.exists(init_pdb): os.remove(init_pdb)
        
        return True, pair_id
    except Exception as e:
        print(f"\n❌ {pair_id} PyRosetta 折叠失败: {e}")
        return False, pair_id

# ================= 4. 主流程控制 =================
def main():
    print("=" * 65)
    print(" 🚀 开始二分类 3D 复合体预测管道 (多进程加速版)")
    print("=" * 65)
    
    # 步骤 1：快速进行模型推理，提取所有概率矩阵
    test_dataset = RnaPeptideDataset(
        os.path.join(SPLIT_DIR, "test_list_59.txt"),
        RNA_DIR, PEP_DIR, TOPO_DIR, LABEL_DIR)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = ZHMolTopoRPI_Network(rna_dim=1280, pep_dim=2560, hidden_dim=128, topo_dim=1300, topo_hidden=64).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
    model.eval()
    
    tasks = []
    print("阶段 1/2: 深度学习特征提取与推理...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="推理进度"):
            pair_id = batch['pair_id'][0]
            final_pdb = os.path.join(OUT_COMPLEX_DIR, f"{pair_id}_complex.pdb")
            if os.path.exists(final_pdb):
                continue
                
            pep_pdb = os.path.join(MONOMER_PEP_DIR, f"{pair_id}_peptide.pdb")
            rna_pdb = os.path.join(MONOMER_RNA_DIR, f"{pair_id}_rna.pdb")
            if not os.path.exists(pep_pdb) or not os.path.exists(rna_pdb):
                continue
                
            rna_feat = batch['feat_rna'].to(DEVICE)
            pep_feat = batch['feat_pep'].to(DEVICE)
            topo_feat = batch['topo_feat'].to(DEVICE)
            mask = batch['mask'][0]
            
            logits = model(rna_feat, pep_feat, topo_feat)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            L_r = int(mask[:, 0].sum().item())
            L_p = int(mask[0, :].sum().item())
            real_probs = probs[:L_r, :L_p]
            
            # 将参数打包，为多进程做准备
            tasks.append((pair_id, real_probs, pep_pdb, rna_pdb, TMP_DIR, OUT_COMPLEX_DIR))
            
    if not tasks:
        print("没有需要处理的新任务。")
        return
        
    print(f"\n阶段 2/2: 并行执行 PyRosetta 折叠 (共 {len(tasks)} 个任务)...")
    
    # 根据CPU核心数自动分配，建议最大并发数为物理核心数（防止过度竞争导致性能下降）
    max_workers = min(os.cpu_count(), len(tasks), 32) 
    success_count = 0
    
    # 使用进程池，并指定初始化函数，确保每个子进程独立初始化 PyRosetta
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_pyrosetta_worker) as executor:
        futures = {executor.submit(process_single_complex, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="多进程 Docking 进度"):
            success, _ = future.result()
            if success:
                success_count += 1
                
    print(f"\n🎉 3D 复合体预测全部完成！成功生成了 {success_count} 个复合物结构。")
    print(f"📁 结果保存在: {OUT_COMPLEX_DIR}")

if __name__ == "__main__":
    main()

