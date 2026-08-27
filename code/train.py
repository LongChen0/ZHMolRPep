import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from dataset import RnaPeptideDataset
from model import ZHMolTopoRPI_Network

def evaluate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            rna = batch['feat_rna'].to(device)
            pep = batch['feat_pep'].to(device)
            topo = batch['topo_feat'].to(device)
            target = batch['contact_map'].to(device)
            mask = batch['mask'].to(device)

            logits = model(rna, pep, topo)
            loss_matrix = criterion(logits, target)
            loss = (loss_matrix * mask).sum() / mask.sum()
            val_loss += loss.item()

            preds = torch.sigmoid(logits)
            all_preds.append(preds[mask.bool()].cpu().numpy())
            all_targets.append(target[mask.bool()].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    pred_binary = (all_preds > 0.5).astype(int)

    precision = precision_score(all_targets, pred_binary, zero_division=0)
    recall = recall_score(all_targets, pred_binary, zero_division=0)
    f1 = f1_score(all_targets, pred_binary, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.0

    return val_loss / len(val_loader), precision, recall, f1, auc

def main():
    BASE_DIR = "/home/zhaolab/long/new_RNA_peptide"
    RNA_DIR = os.path.join(BASE_DIR, "RNA_Peptide_Features/RNA_RiNALMo")
    PEP_DIR = os.path.join(BASE_DIR, "RNA_Peptide_Features/Peptide_ESM2")
    TOPO_DIR = os.path.join(BASE_DIR, "features_3d/ZHMolTopo_3D_monomer/raw_numpy")
    LABEL_DIR = os.path.join(BASE_DIR, "RNA_Peptide_Pairwise/labels")
    SPLIT_DIR = os.path.join(BASE_DIR, "Dataset_Splits")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    EPOCHS = 100              # 改: 加大，让cosine有足够周期
    LR = 5e-5                 # 改: 降低初始学习率 (原来1e-4太激进)
    WEIGHT_DECAY = 5e-4       # 改: 加大weight_decay
    PATIENCE = 15             # 改: 加大patience，给调度器留足时间

    train_dataset = RnaPeptideDataset(
        os.path.join(SPLIT_DIR, "train_list.txt"), RNA_DIR, PEP_DIR, TOPO_DIR, LABEL_DIR)
    val_dataset = RnaPeptideDataset(
        os.path.join(SPLIT_DIR, "val_list.txt"), RNA_DIR, PEP_DIR, TOPO_DIR, LABEL_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ZHMolTopoRPI_Network(rna_dim=1280, pep_dim=2560, hidden_dim=128, topo_dim=1300, topo_hidden=64).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 改: 用 CosineAnnealingLR 替代 ReduceLROnPlateau
    # 这样学习率会平滑下降，不会和EarlyStopping打架
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    pos_weight = torch.tensor([20.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    print("🚀 开始训练: ZHMolTopoRPI 网络 (防过拟合版)...")
    best_val_f1 = 0.0   # 改: 用F1而不是Loss来选最佳模型
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
            rna = batch['feat_rna'].to(DEVICE)
            pep = batch['feat_pep'].to(DEVICE)
            topo = batch['topo_feat'].to(DEVICE)
            target = batch['contact_map'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(rna, pep, topo)
            loss_matrix = criterion(logits, target)
            loss = (loss_matrix * mask).sum() / mask.sum()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()  # Cosine调度: 每个epoch自动step

        val_loss, precision, recall, f1, auc = evaluate(model, val_loader, DEVICE, criterion)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1} | LR: {current_lr:.2e} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        # 改: 用F1作为模型保存标准 (Loss在过拟合时仍可能下降，F1更可靠)
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), "best_zhmoltoporpi_model.pth")
            print(f"🌟 F1提升至 {f1:.4f}，已保存最佳模型！")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"⏹️ {PATIENCE}轮未提升F1，触发Early Stopping。最佳F1: {best_val_f1:.4f}")
                break

if __name__ == "__main__":
    main()

