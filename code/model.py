import torch
import torch.nn as nn

class ResidualBlock2D(nn.Module):
    def __init__(self, dim, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(p=dropout)     # 改: block内也加dropout
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.drop2 = nn.Dropout2d(p=dropout)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        out = self.drop2(out)
        return self.relu(out + residual)

class ZHMolTopoRPI_Network(nn.Module):
    def __init__(self, rna_dim=1280, pep_dim=2560, hidden_dim=64, topo_dim=1300, topo_hidden=32):
        # 改: hidden_dim 128→64, topo_hidden 64→32, 大幅减少参数量
        super().__init__()
        self.rna_proj = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.3)        # 改: 投影后也加dropout
        )
        self.pep_proj = nn.Sequential(
            nn.Linear(pep_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.topo_proj = nn.Sequential(
            nn.Linear(topo_dim, topo_hidden), nn.LayerNorm(topo_hidden), nn.ReLU()
        )

        # 4 * 64 + 32 = 288
        in_channels = hidden_dim * 4 + topo_hidden

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),   # 改: 128→64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),                                    # 改: 0.3→0.4

            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            ResidualBlock2D(64, dropout=0.4),
            ResidualBlock2D(64, dropout=0.4),       # 改: 去掉一个ResBlock，3→2
            nn.Dropout2d(p=0.4),

            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, rna_feat, pep_feat, topo_feat):
        B, L_r, _ = rna_feat.shape
        _, L_p, _ = pep_feat.shape

        r_proj = self.rna_proj(rna_feat)
        p_proj = self.pep_proj(pep_feat)
        t_proj = self.topo_proj(topo_feat)

        r_map = r_proj.unsqueeze(2).expand(B, L_r, L_p, -1)
        p_map = p_proj.unsqueeze(1).expand(B, L_r, L_p, -1)
        t_map = t_proj.view(B, 1, 1, -1).expand(B, L_r, L_p, -1)

        diff_map = r_map - p_map
        mul_map = r_map * p_map

        pair_map = torch.cat([r_map, p_map, diff_map, mul_map, t_map], dim=-1)
        pair_map = pair_map.permute(0, 3, 1, 2)

        logits = self.decoder(pair_map)
        return logits.squeeze(1)

