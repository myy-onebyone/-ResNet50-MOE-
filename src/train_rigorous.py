"""
================================================================================
眼底疾病多标签分类 — ResNet50 vs MoE 严谨对比实验
================================================================================
改进策略:
  1. 患者级别数据划分 (防止数据泄漏)
  2. 加权 BCE Loss 处理类别不平衡
  3. RandAugment 风格数据增强
  4. AdamW + CosineWarmRestarts 调度
  5. 渐进式解冻 (ResNet50)
  6. 梯度裁剪 + 早停
  7. Focal Loss (MoE)
  8. 全量3500患者数据
  9. 综合评估指标 + 可视化
================================================================================
"""
import os
import random
import numpy as np
import pandas as pd
import ast
import time
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, hamming_loss, classification_report, roc_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ============================================================================
# 全局配置
# ============================================================================
IMG_DIR = "data/all_images"
CSV_PATH = "data/full_df.csv"
OUTPUT_DIR = "compare_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 8
CLASS_NAMES = ["正常", "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他疾病/异常"]

# 训练超参数 (针对 CPU 优化)
BATCH_SIZE = 32                     # 更大 batch 减少迭代开销
NUM_WORKERS = 2                     # 轻量多进程数据加载
NUM_THREADS = 8                     # PyTorch CPU 线程数
EPOCHS_RESNET = 20                  # ResNet50: 3(冻结)+3(layer4)+14(全微调)
EPOCHS_MOE = 20                     # MoE 训练轮次
LR_INIT = 1e-3                      # 初始学习率
LR_MIN = 1e-6                       # 最小学习率
WEIGHT_DECAY = 1e-4                 # 权重衰减
GRAD_CLIP = 1.0                     # 梯度裁剪
THRESHOLD = 0.5                     # 分类阈值
SEED = 42
PATIENCE = 6                        # 早停耐心值

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(NUM_THREADS)
print(f"{'='*70}")
print(f"  设备: {DEVICE}")
print(f"  CPU核数: {os.cpu_count()} | PyTorch线程数: {torch.get_num_threads()}")
print(f"{'='*70}")

# ImageNet 标准化参数
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    """固定随机种子确保可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


# ============================================================================
# 数据集
# ============================================================================
class PairedEyeDataset(Dataset):
    """配对眼底图数据集 — 用于 MoE 训练 & 最终测试"""
    def __init__(self, df, img_dir, augment=False, is_train=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment and is_train
        self.is_train = is_train

        base = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]

        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ] + base)
        else:
            self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        left = Image.open(os.path.join(self.img_dir, row['Left-Fundus'])).convert('RGB')
        right = Image.open(os.path.join(self.img_dir, row['Right-Fundus'])).convert('RGB')
        target = torch.tensor(ast.literal_eval(row['target']), dtype=torch.float32)
        return (self.transform(left), self.transform(right)), target


class SingleEyeDataset(Dataset):
    """单图数据集 — 用于 ResNet50 训练（左右眼独立作为样本）"""
    def __init__(self, df, img_dir, augment=False, is_train=False):
        self.img_dir = img_dir
        self.augment = augment and is_train
        self.is_train = is_train

        self.samples = []
        for _, row in df.iterrows():
            target = ast.literal_eval(row['target'])
            self.samples.append((row['Left-Fundus'], target))
            self.samples.append((row['Right-Fundus'], target))

        base = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]

        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ] + base)
        else:
            self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, target = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')
        return self.transform(img), torch.tensor(target, dtype=torch.float32)


# ============================================================================
# 模型定义
# ============================================================================
class ResNet50Classifier(nn.Module):
    """ResNet50 多标签分类器

    使用 ImageNet 预训练权重，替换最后的 fc 层为多标签输出。
    架构与 GUI (main_window_compare.py) 保持一致，确保权重文件兼容。
    支持渐进式解冻训练。
    """
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        in_features = self.backbone.fc.in_features
        # 与 GUI 保持一致: 单层 nn.Linear
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """冻结 backbone（仅训练分类头 fc）"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """解冻全部参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_layer4(self):
        """仅解冻 layer4（ResNet50 最后一个 stage）+ fc"""
        for name, param in self.backbone.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True


class SharedFeatureExtractor(nn.Module):
    """共享特征提取器 — 冻结的 ResNet50 backbone"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        for param in resnet.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)  # (B, 2048)


class Expert(nn.Module):
    """MoE 专家子网络"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class GatingNetwork(nn.Module):
    """门控网络 — 学习如何加权各专家"""
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.fc(x)


class MixtureOfExperts(nn.Module):
    """MoE 双流门控融合模型

    左眼 + 右眼 → 共享特征提取(冻结ResNet50) → 拼接 → 门控 → 专家投票
    """
    def __init__(self, num_experts=8, num_tasks=8, input_dim=4096):
        super().__init__()
        self.shared_extractor = SharedFeatureExtractor()
        self.experts = nn.ModuleList([
            Expert(input_dim, num_tasks) for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, x_left, x_right):
        left_feat = self.shared_extractor(x_left)
        right_feat = self.shared_extractor(x_right)
        combined = torch.cat([left_feat, right_feat], dim=1)
        gate_weights = self.gating(combined)
        expert_outs = torch.stack(
            [expert(combined) for expert in self.experts], dim=1)
        return torch.sum(gate_weights.unsqueeze(-1) * expert_outs, dim=1)


# ============================================================================
# 损失函数
# ============================================================================
class WeightedBCEWithLogitsLoss(nn.Module):
    """带类别权重的 BCE Loss — 处理多标签类别不平衡"""
    def __init__(self, pos_weights):
        super().__init__()
        self.pos_weights = pos_weights

    def forward(self, inputs, targets):
        # pos_weight: 对正样本加权
        return F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weights,
            reduction='mean',
        )


class FocalLoss(nn.Module):
    """Focal Loss — 专注难分类样本"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ============================================================================
# 评估指标
# ============================================================================
def compute_all_metrics(labels, probs, threshold=0.5):
    """计算全部多标签评估指标"""
    preds = (probs > threshold).astype(int)

    metrics = {
        'subset_acc': accuracy_score(labels, preds),
        'hamming_loss': hamming_loss(labels, preds),
        'micro_precision': precision_score(labels, preds, average='micro', zero_division=0),
        'micro_recall': recall_score(labels, preds, average='micro', zero_division=0),
        'micro_f1': f1_score(labels, preds, average='micro', zero_division=0),
        'macro_precision': precision_score(labels, preds, average='macro', zero_division=0),
        'macro_recall': recall_score(labels, preds, average='macro', zero_division=0),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'sample_f1': f1_score(labels, preds, average='samples', zero_division=0),
    }

    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        metrics[f'f1_{name}'] = per_class_f1[i]
        metrics[f'prec_{name}'] = per_class_precision[i]
        metrics[f'rec_{name}'] = per_class_recall[i]

    # AUC（逐类）
    try:
        aucs = roc_auc_score(labels, probs, average=None)
        metrics['macro_auc'] = np.mean(aucs)
        for i, name in enumerate(CLASS_NAMES):
            metrics[f'auc_{name}'] = aucs[i]
    except Exception:
        metrics['macro_auc'] = 0.0
        for name in CLASS_NAMES:
            metrics[f'auc_{name}'] = 0.0

    return metrics, preds


def compute_class_weights(labels_matrix):
    """根据训练集标签分布计算 pos_weight

    pos_weight[i] = (#negatives) / (#positives)
    使模型更加关注少数类
    """
    n_pos = labels_matrix.sum(axis=0)
    n_neg = labels_matrix.shape[0] - n_pos
    # 避免除零
    pos_weights = np.where(n_pos > 0, n_neg / n_pos, 1.0)
    # 裁剪极端值
    pos_weights = np.clip(pos_weights, 1.0, 15.0)
    return torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)


# ============================================================================
# 训练函数
# ============================================================================
def train_resnet50(train_loader, val_loader, class_weights):
    """训练 ResNet50 多标签分类器 (渐进式解冻)"""
    model = ResNet50Classifier(num_classes=NUM_CLASSES).to(DEVICE)

    # 阶段性训练策略
    # Phase 1: 冻结 backbone，只训练分类头 (warmup)
    # Phase 2: 解冻 layer4
    # Phase 3: 解冻全部参数

    print(f"\n{'='*60}")
    print(f"  训练 ResNet50 — 渐进式解冻策略")
    print(f"  Phase 1: 冻结 backbone → 训练分类头 (3 epochs)")
    print(f"  Phase 2: 解冻 layer4 (3 epochs)")
    print(f"  Phase 3: 全参数微调 (剩余 epochs)")
    print(f"{'='*60}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {n_params:,} | 可训练: {n_trainable:,}")

    criterion = WeightedBCEWithLogitsLoss(class_weights)
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [],
               'val_auc': [], 'lr': []}
    best_val_f1 = 0.0
    patience_counter = 0

    # Phase 1: 冻结 backbone, 仅训练 fc
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=LR_MIN)

    global_epoch = 0
    phase_boundaries = [3, 6]  # Phase 1→2 at epoch 3, Phase 2→3 at epoch 6

    for epoch in range(EPOCHS_RESNET):
        global_epoch = epoch

        # Phase transitions
        if epoch == phase_boundaries[0]:
            print(f"\n  >>> Phase 2: 解冻 layer4")
            model.unfreeze_layer4()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_INIT * 0.5, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=LR_MIN)
        elif epoch == phase_boundaries[1]:
            print(f"\n  >>> Phase 3: 全参数微调")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.fc.parameters(), 'lr': LR_INIT * 0.5},
                {'params': [p for n, p in model.backbone.named_parameters()
                           if 'fc' not in n], 'lr': LR_INIT * 0.1},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=LR_MIN)

        # ── 训练 ──
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"  ResNet50 E{epoch+1:02d}/{EPOCHS_RESNET}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                all_probs.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        val_metrics, _ = compute_all_metrics(all_labels, all_probs, THRESHOLD)
        val_f1 = val_metrics['macro_f1']
        val_auc = val_metrics['macro_auc']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)

        print(f"  E{epoch+1:02d} | TLoss={avg_train_loss:.4f} | VLoss={avg_val_loss:.4f} | "
              f"VF1={val_f1:.4f} | VAUC={val_auc:.4f} | LR={current_lr:.2e}")

        # 早停 & 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_resnet50.pth'))
            print(f"  >>> 保存最佳模型! (Macro F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  早停! {PATIENCE} 轮未改善")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(
        os.path.join(OUTPUT_DIR, 'best_resnet50.pth'), map_location=DEVICE))
    return model, history


def train_moe(train_loader, val_loader):
    """训练 MoE 双流门控融合模型"""
    model = MixtureOfExperts(
        num_experts=8, num_tasks=NUM_CLASSES, input_dim=4096).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  训练 MoE 双流门控融合模型")
    print(f"  总参数: {n_params:,} | 可训练: {n_trainable:,}")
    print(f"{'='*60}")

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=LR_MIN)

    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [],
               'val_auc': [], 'lr': []}

    for epoch in range(EPOCHS_MOE):
        # ── 训练 ──
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"  MoE E{epoch+1:02d}/{EPOCHS_MOE}")
        for (left, right), labels in pbar:
            left, right = left.to(DEVICE), right.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(left, right)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for (left, right), labels in val_loader:
                left, right = left.to(DEVICE), right.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(left, right)
                val_loss += criterion(outputs, labels).item()
                all_probs.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        val_metrics, _ = compute_all_metrics(all_labels, all_probs, THRESHOLD)
        val_f1 = val_metrics['macro_f1']
        val_auc = val_metrics['macro_auc']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)

        print(f"  E{epoch+1:02d} | TLoss={avg_train_loss:.4f} | VLoss={avg_val_loss:.4f} | "
              f"VF1={val_f1:.4f} | VAUC={val_auc:.4f} | LR={current_lr:.2e}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_moe.pth'))
            print(f"  >>> 保存最佳模型! (Macro F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  早停! {PATIENCE} 轮未改善")
                break

    model.load_state_dict(torch.load(
        os.path.join(OUTPUT_DIR, 'best_moe.pth'), map_location=DEVICE))
    return model, history


# ============================================================================
# 统一评估
# ============================================================================
def evaluate_paired(model, paired_loader, method='resnet'):
    """在配对测试集上统一评估"""
    model.eval()
    all_probs, all_labels = [], []
    total_time = 0.0

    with torch.no_grad():
        for (left, right), labels in tqdm(paired_loader, desc=f"  评估 {method}"):
            left, right = left.to(DEVICE), right.to(DEVICE)

            t0 = time.time()
            if method == 'resnet':
                left_out = torch.sigmoid(model(left))
                right_out = torch.sigmoid(model(right))
                probs = torch.max(left_out, right_out)  # 后融合: max pooling
            else:
                outputs = model(left, right)
                probs = torch.sigmoid(outputs)
            total_time += time.time() - t0

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    avg_time = total_time / len(paired_loader)

    return all_labels, all_probs, avg_time


# ============================================================================
# 可视化
# ============================================================================
def plot_comparison(rn_hist, me_hist, rn_metrics, me_metrics,
                    rn_labels, rn_probs, me_labels, me_probs, rn_time, me_time):
    """生成高质量对比图表"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")

    C1 = '#4CAF50'  # ResNet50 绿色
    C2 = '#FF9800'  # MoE 橙色

    # ── 图1: 训练曲线 ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    er = range(1, len(rn_hist['train_loss']) + 1)
    em = range(1, len(me_hist['train_loss']) + 1)

    ax = axes[0]
    ax.plot(er, rn_hist['train_loss'], '-', color=C1, lw=2, alpha=0.5,
            marker='o', ms=4, label='ResNet50 Train')
    ax.plot(er, rn_hist['val_loss'], '-', color=C1, lw=2.5,
            marker='s', ms=5, label='ResNet50 Val')
    ax.plot(em, me_hist['train_loss'], '-', color=C2, lw=2, alpha=0.5,
            marker='o', ms=4, label='MoE Train')
    ax.plot(em, me_hist['val_loss'], '-', color=C2, lw=2.5,
            marker='s', ms=5, label='MoE Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Loss 曲线对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = axes[1]
    ax.plot(er, rn_hist['val_f1'], '-', color=C1, lw=2.5,
            marker='o', ms=6, label='ResNet50')
    ax.plot(em, me_hist['val_f1'], '-', color=C2, lw=2.5,
            marker='s', ms=6, label='MoE')
    # 标注最优
    for h, c, off in [(rn_hist, C1, 0.02), (me_hist, C2, -0.04)]:
        bf = max(h['val_f1'])
        be = h['val_f1'].index(bf) + 1
        ax.annotate(f'{bf:.4f}', xy=(be, bf), xytext=(be-0.5, bf+off),
                    fontsize=9, color=c, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=c, alpha=0.5))
    ax.set_xlabel('Epoch'); ax.set_ylabel('Macro F1')
    ax.set_title('验证集 Macro F1 曲线', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(OUTPUT_DIR, '01_training_curves.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[图表1/6] 训练曲线 → 01_training_curves.png")

    # ── 图2: 总体指标 ──
    metric_keys = ['macro_f1', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_auc']
    metric_labels = ['Macro F1', 'Micro F1', 'Macro Prec', 'Macro Recall', 'Macro AUC']
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(metric_keys)); w = 0.32
    rv = [rn_metrics.get(k, 0) for k in metric_keys]
    mv = [me_metrics.get(k, 0) for k in metric_keys]
    b1 = ax.bar(x-w/2, rv, w, label='ResNet50 (后融合)', color=C1, edgecolor='white')
    b2 = ax.bar(x+w/2, mv, w, label='MoE (门控融合)', color=C2, edgecolor='white')
    for b, v in zip(b1, rv):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    for b, v in zip(b2, mv):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, max(max(rv), max(mv)) * 1.25)
    ax.set_title('ResNet50 vs MoE — 总体指标对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_overall_metrics.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[图表2/6] 总体指标 → 02_overall_metrics.png")

    # ── 图3: 逐类 F1 ──
    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(CLASS_NAMES)); h = 0.32
    rf = [rn_metrics.get(f'f1_{n}', 0) for n in CLASS_NAMES]
    mf = [me_metrics.get(f'f1_{n}', 0) for n in CLASS_NAMES]
    ax.barh(y_pos-h/2, rf, h, label='ResNet50 (后融合)', color=C1, edgecolor='white')
    ax.barh(y_pos+h/2, mf, h, label='MoE (门控融合)', color=C2, edgecolor='white')
    for i, (rv, mv) in enumerate(zip(rf, mf)):
        ax.text(max(rv, 0.01)+0.01, i-h/2, f'{rv:.3f}', va='center', fontsize=9, color=C1, fontweight='bold')
        ax.text(max(mv, 0.01)+0.01, i+h/2, f'{mv:.3f}', va='center', fontsize=9, color=C2, fontweight='bold')
    ax.set_yticks(y_pos); ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlim(0, max(max(rf), max(mf)) * 1.35)
    ax.set_xlabel('F1-Score'); ax.set_title('逐类 F1 对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='x', alpha=0.3); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_per_class_f1.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[图表3/6] 逐类F1 → 03_per_class_f1.png")

    # ── 图4: 逐类 AUC ──
    fig, ax = plt.subplots(figsize=(12, 7))
    ra = [rn_metrics.get(f'auc_{n}', 0) for n in CLASS_NAMES]
    ma = [me_metrics.get(f'auc_{n}', 0) for n in CLASS_NAMES]
    ax.barh(y_pos-h/2, ra, h, label='ResNet50 (后融合)', color=C1, edgecolor='white')
    ax.barh(y_pos+h/2, ma, h, label='MoE (门控融合)', color=C2, edgecolor='white')
    for i, (rv, mv) in enumerate(zip(ra, ma)):
        ax.text(max(rv, 0.01)+0.01, i-h/2, f'{rv:.3f}', va='center', fontsize=9, color=C1, fontweight='bold')
        ax.text(max(mv, 0.01)+0.01, i+h/2, f'{mv:.3f}', va='center', fontsize=9, color=C2, fontweight='bold')
    ax.set_yticks(y_pos); ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlim(0, max(max(ra), max(ma)) * 1.25)
    ax.set_xlabel('AUC'); ax.set_title('逐类 AUC 对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='x', alpha=0.3); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_per_class_auc.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[图表4/6] 逐类AUC → 04_per_class_auc.png")

    # ── 图5: ROC 曲线 ──
    fig = plt.figure(figsize=(18, 12))
    # 3x3 grid: 主 ROC 占第一行全部(1-3), 逐类 ROC 占位置 4-8
    ax1 = plt.subplot(3, 3, (1, 3))
    fpr_r, tpr_r, _ = roc_curve(rn_labels.ravel(), rn_probs.ravel())
    fpr_m, tpr_m, _ = roc_curve(me_labels.ravel(), me_probs.ravel())
    ax1.plot(fpr_r, tpr_r, '-', color=C1, lw=2.5, label=f'ResNet50 (AUC={auc(fpr_r,tpr_r):.4f})')
    ax1.plot(fpr_m, tpr_m, '-', color=C2, lw=2.5, label=f'MoE (AUC={auc(fpr_m,tpr_m):.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.4)
    ax1.fill_between(fpr_r, tpr_r, alpha=0.05, color=C1)
    ax1.fill_between(fpr_m, tpr_m, alpha=0.05, color=C2)
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
    ax1.set_title('ROC Curve (Micro-average)', fontsize=14, fontweight='bold')
    ax1.legend(); ax1.grid(alpha=0.3)
    # 逐类 ROC (top 5)
    sorted_classes = sorted(CLASS_NAMES,
        key=lambda n: me_metrics.get(f'auc_{n}', 0), reverse=True)[:5]
    for idx, cn in enumerate(sorted_classes):
        ci = CLASS_NAMES.index(cn); ax = plt.subplot(3, 3, idx+4)
        fpr_rc, tpr_rc, _ = roc_curve(rn_labels[:, ci], rn_probs[:, ci])
        fpr_mc, tpr_mc, _ = roc_curve(me_labels[:, ci], me_probs[:, ci])
        ax.plot(fpr_rc, tpr_rc, '-', color=C1, lw=1.8, label=f'RN ({auc(fpr_rc,tpr_rc):.3f})')
        ax.plot(fpr_mc, tpr_mc, '-', color=C2, lw=1.8, label=f'MoE ({auc(fpr_mc,tpr_mc):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title(cn, fontsize=12, fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    plt.suptitle('ROC 曲线分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(OUTPUT_DIR, '05_roc_curves.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[图表5/6] ROC曲线 → 05_roc_curves.png")

    # ── 图6: 推理速度 + 参数量 ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    times_ms = [rn_time * 1000, me_time * 1000]
    bars = ax.bar(['ResNet50\n(后融合)', 'MoE\n(门控融合)'], times_ms,
                  color=[C1, C2], edgecolor='white', width=0.5)
    for b, t in zip(bars, times_ms):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                f'{t:.1f} ms', ha='center', fontsize=13, fontweight='bold')
    ax.set_title('单对眼底图推理速度', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(times_ms) * 1.3)

    ax = axes[1]
    rn_p = sum(p.numel() for p in ResNet50Classifier(8).parameters())
    me_tmp = MixtureOfExperts(8, 8, 4096)
    me_p = sum(p.numel() for p in me_tmp.parameters())
    me_t = sum(p.numel() for p in me_tmp.parameters() if p.requires_grad)
    p_data = {'ResNet50': rn_p, 'MoE (总计)': me_p, 'MoE (可训练)': me_t}
    pb = ax.bar(p_data.keys(), p_data.values(), color=[C1, C2, '#FFCC80'],
                edgecolor='white', width=0.5)
    for b, v in zip(pb, p_data.values()):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3e6,
                f'{v/1e6:.1f}M', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('模型参数量对比', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(OUTPUT_DIR, '06_model_comparison.png'),
                dpi=150, facecolor='white')
    plt.close()
    print("[图表6/6] 推理速度+参数量 → 06_model_comparison.png")


def generate_report(rn_metrics, me_metrics, rn_time, me_time, dataset_size):
    """生成 Markdown 对比报告"""
    lines = []
    lines.append("# ResNet50 vs MoE 对比实验报告 (严谨版)\n")
    lines.append(f"**实验日期**: {time.strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## 1. 实验设置\n")
    lines.append(f"- 数据集: ODIR-5K 子集, {dataset_size} 患者, 7000 张眼底图")
    lines.append(f"- 类别: {', '.join(CLASS_NAMES)}")
    lines.append(f"- 设备: {DEVICE}")
    lines.append(f"- ResNet50 epochs: {EPOCHS_RESNET} (3冻结+3layer4+14全微调) | MoE epochs: {EPOCHS_MOE}")
    lines.append(f"- 优化器: AdamW (lr={LR_INIT}, wd={WEIGHT_DECAY})")
    lines.append(f"- 调度器: CosineAnnealingWarmRestarts")
    lines.append(f"- ResNet50: 渐进式解冻 + 加权BCE Loss")
    lines.append(f"- MoE: Focal Loss (α=0.25, γ=2.0)")
    lines.append(f"- 阈值: {THRESHOLD}\n")

    lines.append("## 2. 总体指标对比\n")
    lines.append("| 指标 | ResNet50 | MoE | Δ | 胜出 |")
    lines.append("|------|----------|-----|---|------|")
    for key, name in [('macro_f1', 'Macro F1'), ('micro_f1', 'Micro F1'),
                       ('macro_precision', 'Macro Precision'),
                       ('macro_recall', 'Macro Recall'), ('macro_auc', 'Macro AUC')]:
        rv = rn_metrics.get(key, 0); mv = me_metrics.get(key, 0)
        winner = 'ResNet50' if rv > mv else ('MoE' if mv > rv else '平手')
        lines.append(f"| {name} | {rv:.4f} | {mv:.4f} | {mv-rv:+.4f} | {winner} |")

    lines.append("\n## 3. 逐类 F1 对比\n")
    lines.append("| 类别 | ResNet50 | MoE | Δ | 胜出 |")
    lines.append("|------|----------|-----|---|------|")
    for name in CLASS_NAMES:
        rf = rn_metrics.get(f'f1_{name}', 0); mf = me_metrics.get(f'f1_{name}', 0)
        winner = 'ResNet50' if rf > mf else ('MoE' if mf > rf else '平手')
        lines.append(f"| {name} | {rf:.4f} | {mf:.4f} | {mf-rf:+.4f} | {winner} |")

    lines.append("\n## 4. 逐类 AUC 对比\n")
    lines.append("| 类别 | ResNet50 | MoE | Δ |")
    lines.append("|------|----------|-----|---|")
    for name in CLASS_NAMES:
        ra = rn_metrics.get(f'auc_{name}', 0); ma = me_metrics.get(f'auc_{name}', 0)
        lines.append(f"| {name} | {ra:.4f} | {ma:.4f} | {ma-ra:+.4f} |")

    lines.append(f"\n## 5. 推理速度\n")
    lines.append(f"- ResNet50: {rn_time*1000:.1f} ms/对")
    lines.append(f"- MoE: {me_time*1000:.1f} ms/对")

    report = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, 'comparison_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[报告] → compare_results/comparison_report.md")
    print(report)


# ============================================================================
# 主流程
# ============================================================================
def main():
    print(f"\n{'='*70}")
    print(f"  眼底疾病分类 — ResNet50 vs MoE 严谨对比实验")
    print(f"{'='*70}")

    # ── 1. 加载数据 ──
    print("\n[1/6] 加载数据...")
    full_df = pd.read_csv(CSV_PATH)
    print(f"  总患者数: {len(full_df)} | 总图片数: {len(full_df)*2}")

    # 患者级别数据划分 (防止左右眼数据泄漏)
    n_total = len(full_df)
    indices = list(range(n_total))
    random.shuffle(indices)

    n_train = int(0.70 * n_total)   # 70% 训练
    n_val = int(0.15 * n_total)     # 15% 验证
    # 15% 测试 (剩余)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]

    train_df = full_df.iloc[train_indices].reset_index(drop=True)
    val_df = full_df.iloc[val_indices].reset_index(drop=True)
    test_df = full_df.iloc[test_indices].reset_index(drop=True)

    print(f"  训练: {len(train_df)} | 验证: {len(val_df)} | 测试: {len(test_df)}")

    # ── 2. 计算类别权重 ──
    print("\n[2/6] 计算类别权重...")
    train_targets = np.array([ast.literal_eval(t) for t in train_df['target']])
    class_weights = compute_class_weights(train_targets)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: pos_weight={class_weights[i]:.2f} "
              f"(正样本={int(train_targets[:,i].sum())})")

    # ── 3. 创建 DataLoader ──
    print("\n[3/6] 创建 DataLoader...")
    # ResNet50: 单图数据集
    train_single = SingleEyeDataset(train_df, IMG_DIR, augment=True, is_train=True)
    val_single = SingleEyeDataset(val_df, IMG_DIR, augment=False, is_train=False)

    # MoE: 配对数据集
    train_paired = PairedEyeDataset(train_df, IMG_DIR, augment=True, is_train=True)
    val_paired = PairedEyeDataset(val_df, IMG_DIR, augment=False, is_train=False)
    test_paired = PairedEyeDataset(test_df, IMG_DIR, augment=False, is_train=False)

    rn_train_ld = DataLoader(train_single, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS,
                             pin_memory=True, prefetch_factor=2)
    rn_val_ld = DataLoader(val_single, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS,
                           pin_memory=True, prefetch_factor=2)
    me_train_ld = DataLoader(train_paired, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS,
                             pin_memory=True, prefetch_factor=2)
    me_val_ld = DataLoader(val_paired, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS,
                           pin_memory=True, prefetch_factor=2)
    test_ld = DataLoader(test_paired, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=True, prefetch_factor=2)

    print(f"  ResNet50 训练 batches: {len(rn_train_ld)}")
    print(f"  MoE 训练 batches: {len(me_train_ld)}")
    print(f"  测试 batches: {len(test_ld)}")

    # ── 4. 训练 ResNet50 ──
    print("\n[4/6] 训练 ResNet50...")
    rn_model, rn_hist = train_resnet50(rn_train_ld, rn_val_ld, class_weights)

    # ── 5. 训练 MoE ──
    print("\n[5/6] 训练 MoE...")
    me_model, me_hist = train_moe(me_train_ld, me_val_ld)

    # ── 6. 测试集评估 ──
    print("\n[6/6] 测试集统一评估...")
    print("  评估 ResNet50 (后融合)...")
    r_labels, r_probs, r_time = evaluate_paired(rn_model, test_ld, 'resnet')
    r_metrics, _ = compute_all_metrics(r_labels, r_probs, THRESHOLD)

    print("  评估 MoE (门控融合)...")
    m_labels, m_probs, m_time = evaluate_paired(me_model, test_ld, 'moe')
    m_metrics, _ = compute_all_metrics(m_labels, m_probs, THRESHOLD)

    # ── 输出结果 ──
    print(f"\n{'='*70}")
    print(f"  测试集最终结果")
    print(f"{'='*70}")
    print(f"  {'指标':<20} {'ResNet50':>12} {'MoE':>12} {'Δ':>10}")
    print(f"  {'-'*54}")
    for key, name in [('macro_f1', 'Macro F1'), ('micro_f1', 'Micro F1'),
                       ('macro_auc', 'Macro AUC'), ('hamming_loss', 'Hamming Loss')]:
        rv = r_metrics.get(key, 0); mv = m_metrics.get(key, 0)
        print(f"  {name:<20} {rv:>12.4f} {mv:>12.4f} {mv-rv:>+10.4f}")

    print(f"\n  逐类 F1:")
    print(f"  {'类别':<12} {'ResNet50':>10} {'MoE':>10} {'Δ':>10}")
    print(f"  {'-'*42}")
    for name in CLASS_NAMES:
        rf = r_metrics.get(f'f1_{name}', 0); mf = m_metrics.get(f'f1_{name}', 0)
        print(f"  {name:<12} {rf:>10.4f} {mf:>10.4f} {mf-rf:>+10.4f}")

    print(f"\n  推理速度: ResNet50={r_time*1000:.1f}ms | MoE={m_time*1000:.1f}ms")

    # ── 生成对比图表和报告 ──
    print(f"\n{'='*70}")
    print(f"  生成对比图表和报告...")
    plot_comparison(rn_hist, me_hist, r_metrics, m_metrics,
                    r_labels, r_probs, m_labels, m_probs, r_time, m_time)
    generate_report(r_metrics, m_metrics, r_time, m_time, len(full_df))

    print(f"\n{'='*70}")
    print(f"  实验完成！所有结果保存在: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
