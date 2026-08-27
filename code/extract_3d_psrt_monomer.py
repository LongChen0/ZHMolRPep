import os
import glob
import numpy as np
from Bio.PDB import PDBParser
from psrt import PH
from tqdm import tqdm
import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

# ==========================================
# 全局配置参数
# ==========================================
CONFIG = {
    'encoder': 'ZHMolTopo_3D_monomer',  # 新目录名，与旧的复合物特征区分开!
    'rna_pdb_dir': '/home/zhaolab/long/new_RNA_peptide/RNA_Peptide_Pairwise/monomer_pdbs/rnas',
    'pep_pdb_dir': '/home/zhaolab/long/new_RNA_peptide/RNA_Peptide_Pairwise/monomer_pdbs/peptides',
    # --- 极其关键的物理参数 ---
    'max_dimension': 2,
    'num_filtration_points': 50,
    'filtration_max_dist': 15.0,
}


def extract_points_from_pdb(pdb_path, target_atoms):
    """
    通用点云提取器：从指定的 PDB 中提取目标原子坐标
    target_atoms: list, 例如 ['P', "C4'"] 或 ['CA']
    """
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("tmp", pdb_path)
    except Exception:
        return None

    coords = []
    for model in struct:
        for chain in model:
            for res in chain:
                # 跳过杂原子
                if res.id[0] != ' ':
                    continue
                for atom_name in target_atoms:
                    if atom_name in res:
                        coords.append(res[atom_name].get_coord())
                        break  # 只要取到一个就跳出，比如优先取P，没有再取C4'
    if not coords or len(coords) < 3:
        return None
    return np.array(coords, dtype=np.float64)


def compute_and_save_ph_features(points, pair_id, tag, save_dir, config):
    """
    计算拓扑特征并保存
    tag: 'rna' 或 'pep'
    """
    max_dim = config['max_dimension']
    max_dist = config['filtration_max_dist']
    num_points = config['num_filtration_points']
    specific_filtration = np.linspace(0, max_dist, num_points)

    ph = PH(
        points=points,
        max_dimension=max_dim,
        max_edge_length=max_dist,
        specific_filtration=specific_filtration
    )

    # 计算所有特征曲线
    alphas, b_curves = ph.betti_curves()
    f_curves_dict = ph.compute_f_vector_curves()
    h_curves_dict = ph.compute_h_vector_curves()
    facet_curves_dict = ph.facet_curves()

    # 保存每个维度的特征
    for d in range(max_dim + 1):
        b_curve = b_curves.get(d, np.zeros(num_points))
        np.save(os.path.join(save_dir, f"{pair_id}_{tag}_betti{d}.npy"), b_curve)

        f_curve = f_curves_dict.get(d, np.zeros(num_points))
        np.save(os.path.join(save_dir, f"{pair_id}_{tag}_f{d}.npy"), f_curve)

        facet_curve = facet_curves_dict.get(d, np.zeros(num_points))
        np.save(os.path.join(save_dir, f"{pair_id}_{tag}_facet{d}.npy"), facet_curve)

    for d in range(max_dim + 2):
        h_curve = h_curves_dict.get(d, np.zeros(num_points))
        np.save(os.path.join(save_dir, f"{pair_id}_{tag}_h{d}.npy"), h_curve)


def run_monomer_featurization(config):
    print("\n" + "=" * 65)
    print(f" 🧬 启动单体 3D 结构拓扑特征提取 (防数据泄露版)")
    print(f" - 扫描半径: 0.0 Å -> {config['filtration_max_dist']} Å")
    print("=" * 65)

    save_dir = os.path.join("/home/zhaolab/long/new_RNA_peptide/features_3d", config['encoder'], "raw_numpy")
    os.makedirs(save_dir, exist_ok=True)

    # 扫描单体 RNA 结构文件，获取所有的 pair_id
    rna_pdbs = glob.glob(os.path.join(config['rna_pdb_dir'], "*_rna.pdb"))
    pair_ids = [os.path.basename(f).replace("_rna.pdb", "") for f in rna_pdbs]

    if not pair_ids:
        print(f"[!] 在 {config['rna_pdb_dir']} 中未找到文件！")
        return

    print(f"[*] 成功扫描到 {len(pair_ids)} 个预测的单体结构，准备提取特征...")

    success_count = 0
    fail_count = 0
    failed_ids = []

    for pair_id in tqdm(pair_ids, desc="提取单体拓扑特征"):
        rna_pdb = os.path.join(config['rna_pdb_dir'], f"{pair_id}_rna.pdb")
        pep_pdb = os.path.join(config['pep_pdb_dir'], f"{pair_id}_peptide.pdb")

        # 跳过已经处理过的
        if os.path.exists(os.path.join(save_dir, f"{pair_id}_rna_h3.npy")) and \
                os.path.exists(os.path.join(save_dir, f"{pair_id}_pep_h3.npy")):
            success_count += 1
            continue

        if not os.path.exists(rna_pdb) or not os.path.exists(pep_pdb):
            fail_count += 1
            failed_ids.append(pair_id)
            continue

        try:
            # 1. 提取 RNA 骨架点云 (P 原子优先，没有则退而求其次取 C4')
            rna_points = extract_points_from_pdb(rna_pdb, ['P', "C4'"])
            # 2. 提取多肽主链点云 (CA 原子)
            pep_points = extract_points_from_pdb(pep_pdb, ['CA'])

            if rna_points is None or pep_points is None:
                raise ValueError("坐标点云提取失败")

            # 3. 分别计算并保存两者的拓扑特征
            compute_and_save_ph_features(rna_points, pair_id, "rna", save_dir, config)
            compute_and_save_ph_features(pep_points, pair_id, "pep", save_dir, config)

            success_count += 1
        except Exception as e:
            fail_count += 1
            failed_ids.append(pair_id)

    print("\n" + "=" * 65)
    print(f"✅ 单体 3D 特征提取完成！")
    print(f" - 成功提取: {success_count} 个复合物")
    print(f" - 提取失败: {fail_count} 个复合物")
    print(f"📁 特征库已封存至: {save_dir}")
    if fail_count > 0:
        print(f"⚠️ 以下复合物提取失败:")
        print(", ".join(failed_ids[:10]) + ("..." if len(failed_ids) > 10 else ""))


if __name__ == "__main__":
    run_monomer_featurization(CONFIG)
