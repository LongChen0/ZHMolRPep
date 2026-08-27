import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


class RnaPeptideDataset(Dataset):
    def __init__(self, split_file, rna_feat_dir, pep_feat_dir, topo_feat_dir, label_dir, max_rna_len=500,
                 max_pep_len=50):
        self.max_rna_len = max_rna_len
        self.max_pep_len = max_pep_len
        self.rna_feat_dir = rna_feat_dir
        self.pep_feat_dir = pep_feat_dir
        self.topo_feat_dir = topo_feat_dir
        self.label_dir = label_dir

        # 定义需要加载的拓扑特征组件
        self.topo_components = [
            'betti0', 'betti1', 'betti2',
            'f0', 'f1', 'f2',
            'facet0', 'facet1', 'facet2',
            'h0', 'h1', 'h2', 'h3'
        ]
        self.topo_points = 50  # 扫描刻度

        with open(split_file, 'r') as f:
            self.pair_ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, idx):
        pair_id = self.pair_ids[idx]

        # ================= 1. 读取 LLM 特征 =================
        rna_data = torch.load(os.path.join(self.rna_feat_dir, f"{pair_id}.pt"), weights_only=True)
        pep_data = torch.load(os.path.join(self.pep_feat_dir, f"{pair_id}.pt"), weights_only=True)

        feat_rna = rna_data['feat_rna']
        feat_pep = pep_data['feat_peptide']

        L_r, _ = feat_rna.shape
        L_p, _ = feat_pep.shape

        # ================= 2. 读取 3D PSRT 单体拓扑特征 (防数据泄露) =================
        topo_list = []
        # 分别读取 RNA 单体和 Peptide 单体的拓扑特征
        for tag in ['rna', 'pep']:
            for comp in self.topo_components:
                npy_path = os.path.join(self.topo_feat_dir, f"{pair_id}_{tag}_{comp}.npy")
                if os.path.exists(npy_path):
                    arr = np.load(npy_path)
                else:
                    arr = np.zeros(self.topo_points, dtype=np.float32)
                topo_list.append(torch.tensor(arr, dtype=torch.float32))

        # 拼接成 2 * (13 * 50) = 650 维的全局拓扑向量
        topo_feat = torch.cat(topo_list, dim=0)

        # ================= 3. 读取 JSON 标签 =================
        with open(os.path.join(self.label_dir, f"{pair_id}_label.json"), 'r') as f:
            label_data = json.load(f)

        contact_map = torch.zeros((L_r, L_p), dtype=torch.float32)
        for contact in label_data['contacts']:
            r_idx = contact['r_idx'] - 1
            p_idx = contact['p_idx'] - 1
            if 0 <= r_idx < L_r and 0 <= p_idx < L_p:
                contact_map[r_idx, p_idx] = 1.0

        # ================= 4. 动态 Padding =================
        pad_r = self.max_rna_len - L_r
        pad_p = self.max_pep_len - L_p

        feat_rna_padded = F.pad(feat_rna, (0, 0, 0, pad_r))
        feat_pep_padded = F.pad(feat_pep, (0, 0, 0, pad_p))
        contact_map_padded = F.pad(contact_map, (0, pad_p, 0, pad_r))

        mask = torch.zeros((self.max_rna_len, self.max_pep_len), dtype=torch.float32)
        mask[:L_r, :L_p] = 1.0

        return {
            'pair_id': pair_id,
            'feat_rna': feat_rna_padded,
            'feat_pep': feat_pep_padded,
            'topo_feat': topo_feat,  # 650维单体拼接拓扑
            'contact_map': contact_map_padded,
            'mask': mask
        }
