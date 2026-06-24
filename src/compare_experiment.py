"""
眼底疾病多标签分类 — 对比实验
ResNet50 (后融合) vs MoE 双流门控融合

两种方法使用相同的数据集划分，公平对比。
"""
import os
import random
import numpy as np
import pandas as pd
import ast
import time
from collections import Counter

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, hamming_loss, classification_report,
    roc_curve, auc, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹窗
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 配置 ====================
IMG_DIR = "data/all_images"
CSV_PATH = "data/full_df.csv"
OUTPUT_DIR = "compare_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

NUM_CLASSES = 8
CLASS_NAMES = ["正常", "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他疾病/异常"]

# 训练参数（适配笔记本 CPU — 预计 2-3 小时完成）
BATCH_SIZE = 8
NUM_WORKERS = 0       # CPU 设 0 避免多进程开销
EPOCHS = 10           # 10 epoch 足够看出趋势
LR = 0.001
THRESHOLD = 0.5
SEED = 42
USE_SUBSET = True     # 用 2500 样本子集加速训练
SUBSET_SIZE = 2500

def set_seed(seed=42):
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


# ==================== 配对数据集 ====================
class PairedEyeDataset(Dataset):
    """
    每个患者返回左右眼两张图 + 多标签向量。
    用于 MoE 训练 & 统一测试。
    """
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment

        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ] + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        left_img = Image.open(
            os.path.join(self.img_dir, row['Left-Fundus'])).convert('RGB')
        right_img = Image.open(
            os.path.join(self.img_dir, row['Right-Fundus'])).convert('RGB')

        left_tensor = self.transform(left_img)
        right_tensor = self.transform(right_img)

        target = torch.tensor(
            ast.literal_eval(row['target']), dtype=torch.float32)

        return (left_tensor, right_tensor), target


class SingleEyeDataset(Dataset):
    """
    单张眼底图数据集（每张图为独立样本）。
    用于 ResNet50 训练。
    """
    def __init__(self, df, img_dir, augment=False):
        self.img_dir = img_dir
        self.augment = augment

        # 将左右眼拆成两个独立样本
        self.samples = []
        for _, row in df.iterrows():
            target = ast.literal_eval(row['target'])
            self.samples.append((row['Left-Fundus'], target))
            self.samples.append((row['Right-Fundus'], target))

        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ] + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, target = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')
        return self.transform(img), torch.tensor(target, dtype=torch.float32)


# ==================== ResNet50 模型 ====================
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ==================== MoE 模型定义 ====================
class SharedFeatureExtractor(nn.Module):
    """共享特征提取器（冻结ResNet50）"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        for param in resnet.parameters():
            param.requires_grad = False   # 冻结，节省训练时间
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)  # (B, 2048)


class Expert(nn.Module):
    """专家子网络"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class GatingNetwork(nn.Module):
    """门控网络 — 学习如何加权各专家"""
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_experts), nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


class MixtureOfExperts(nn.Module):
    """MoE 双流模型：左眼+右眼 → 共享特征 → 门控 → 专家投票"""
    def __init__(self, num_experts=8, num_tasks=8, input_dim=4096):
        super().__init__()
        self.shared_extractor = SharedFeatureExtractor()
        self.experts = nn.ModuleList(
            [Expert(input_dim, num_tasks) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, x_left, x_right):
        left_feat = self.shared_extractor(x_left)    # (B, 2048)
        right_feat = self.shared_extractor(x_right)  # (B, 2048)
        combined = torch.cat([left_feat, right_feat], dim=1)  # (B, 4096)

        gate_weights = self.gating(combined)  # (B, num_experts)
        expert_outs = torch.stack(
            [expert(combined) for expert in self.experts], dim=1)  # (B, E, 8)
        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outs, dim=1)
        return output


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ==================== 评估指标 ====================
def compute_all_metrics(labels, probs, threshold=0.5):
    """计算全部评估指标"""
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

    # 逐类 F1
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        metrics[f'f1_{name}'] = per_class_f1[i]

    # AUC（逐类）
    try:
        aucs = roc_auc_score(labels, probs, average=None)
        metrics['macro_auc'] = np.mean(aucs)
        for i, name in enumerate(CLASS_NAMES):
            metrics[f'auc_{name}'] = aucs[i]
    except Exception:
        metrics['macro_auc'] = 0

    return metrics, preds


# ==================== ResNet50 训练 ====================
def train_resnet50(train_loader, val_loader):
    """训练 ResNet50（单图输入）"""
    model = ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    print("\n" + "=" * 60)
    print("训练 ResNet50 (单图分类器)")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # ---- 训练 ----
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- 验证 ----
        model.eval()
        val_loss = 0
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
        val_f1 = f1_score(all_labels, (all_probs > 0.5).astype(int),
                          average='macro', zero_division=0)

        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)

        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Macro F1: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_resnet50.pth'))
            print(f"  >>> 保存最佳模型")

    return model, history


# ==================== MoE 训练 ====================
def train_moe(train_loader, val_loader, epochs=EPOCHS):
    """训练 MoE 双流门控模型"""
    model = MixtureOfExperts(
        num_experts=8, num_tasks=NUM_CLASSES, input_dim=4096).to(DEVICE)
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    print("\n" + "=" * 60)
    print("训练 MoE 双流门控融合模型")
    print("=" * 60)

    for epoch in range(epochs):
        # ---- 训练 ----
        model.train()
        train_loss = 0
        for (left, right), labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            left, right = left.to(DEVICE), right.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(left, right)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- 验证 ----
        model.eval()
        val_loss = 0
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
        val_f1 = f1_score(all_labels, (all_probs > 0.5).astype(int),
                          average='macro', zero_division=0)

        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)

        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Macro F1: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_moe.pth'))
            print(f"  >>> 保存最佳模型")

    return model, history


# ==================== 统一评估（配对测试） ====================
def evaluate_paired(model, paired_loader, method='resnet'):
    """
    在配对测试集上评估模型。
    method='resnet' → 单图推理 + 后融合max
    method='moe'    → 双流直接推理
    """
    model.eval()
    all_probs, all_labels = [], []
    inference_times = []

    with torch.no_grad():
        for (left, right), labels in tqdm(paired_loader, desc=f"评估 {method}"):
            left, right = left.to(DEVICE), right.to(DEVICE)

            t0 = time.time()
            if method == 'resnet':
                # 分别推理左右眼 → Sigmoid → max 融合
                left_out = torch.sigmoid(model(left))
                right_out = torch.sigmoid(model(right))
                probs = torch.max(left_out, right_out)       # 后融合
            else:
                # MoE 双流推理
                outputs = model(left, right)
                probs = torch.sigmoid(outputs)               # 已经融合
            t1 = time.time()

            inference_times.append(t1 - t0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    avg_time = np.mean(inference_times)

    return all_labels, all_probs, avg_time


# ==================== 可视化 ====================
def plot_comparison(resnet_metrics, moe_metrics, resnet_probs, moe_probs,
                    resnet_labels, moe_labels, resnet_history, moe_history):
    """生成对比图表"""
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12

    # ── 图1：训练曲线对比 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(resnet_history['train_loss'], 'b-', label='ResNet50 Train', alpha=0.7)
    axes[0].plot(resnet_history['val_loss'], 'b--', label='ResNet50 Val', alpha=0.7)
    axes[0].plot(moe_history['train_loss'], 'r-', label='MoE Train', alpha=0.7)
    axes[0].plot(moe_history['val_loss'], 'r--', label='MoE Val', alpha=0.7)
    axes[0].set_title('Loss 曲线对比')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(resnet_history['val_f1'], 'b-o', label='ResNet50', alpha=0.7)
    axes[1].plot(moe_history['val_f1'], 'r-s', label='MoE', alpha=0.7)
    axes[1].set_title('验证集 Macro F1 曲线')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_training_curves.png'), dpi=150)
    plt.close()
    print("[图表] 训练曲线 → 01_training_curves.png")

    # ── 图2：总体指标对比柱状图 ──
    metric_names = ['macro_f1', 'micro_f1', 'macro_precision',
                    'macro_recall', 'macro_auc']
    metric_labels = ['Macro F1', 'Micro F1', 'Macro Precision',
                     'Macro Recall', 'Macro AUC']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_names))
    width = 0.35

    resnet_vals = [resnet_metrics.get(m, 0) for m in metric_names]
    moe_vals = [moe_metrics.get(m, 0) for m in metric_names]

    bars1 = ax.bar(x - width/2, resnet_vals, width, label='ResNet50 (后融合)',
                   color='#4CAF50', edgecolor='white')
    bars2 = ax.bar(x + width/2, moe_vals, width, label='MoE (门控融合)',
                   color='#FF9800', edgecolor='white')

    # 数值标注
    for bar, val in zip(bars1, resnet_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, moe_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('ResNet50 vs MoE — 总体指标对比')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_overall_metrics.png'), dpi=150)
    plt.close()
    print("[图表] 总体指标 → 02_overall_metrics.png")

    # ── 图3：逐类 F1 对比 ──
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    resnet_f1s = [resnet_metrics.get(f'f1_{name}', 0) for name in CLASS_NAMES]
    moe_f1s = [moe_metrics.get(f'f1_{name}', 0) for name in CLASS_NAMES]

    ax.bar(x - width/2, resnet_f1s, width, label='ResNet50 (后融合)',
           color='#4CAF50', edgecolor='white')
    ax.bar(x + width/2, moe_f1s, width, label='MoE (门控融合)',
           color='#FF9800', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('F1-Score')
    ax.set_title('ResNet50 vs MoE — 逐类 F1 对比')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_per_class_f1.png'), dpi=150)
    plt.close()
    print("[图表] 逐类F1 → 03_per_class_f1.png")

    # ── 图4：逐类 AUC 对比 ──
    resnet_aucs = [resnet_metrics.get(f'auc_{name}', 0) for name in CLASS_NAMES]
    moe_aucs = [moe_metrics.get(f'auc_{name}', 0) for name in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, resnet_aucs, width, label='ResNet50 (后融合)',
           color='#4CAF50', edgecolor='white')
    ax.bar(x + width/2, moe_aucs, width, label='MoE (门控融合)',
           color='#FF9800', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('AUC')
    ax.set_title('ResNet50 vs MoE — 逐类 AUC 对比')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_per_class_auc.png'), dpi=150)
    plt.close()
    print("[图表] 逐类AUC → 04_per_class_auc.png")

    # ── 图5：ROC 曲线（微平均） ──
    fig, ax = plt.subplots(figsize=(8, 8))

    # ResNet50 micro-average ROC
    fpr_r, tpr_r, _ = roc_curve(resnet_labels.ravel(), resnet_probs.ravel())
    auc_r = auc(fpr_r, tpr_r)

    # MoE micro-average ROC
    fpr_m, tpr_m, _ = roc_curve(moe_labels.ravel(), moe_probs.ravel())
    auc_m = auc(fpr_m, tpr_m)

    ax.plot(fpr_r, tpr_r, 'b-', label=f'ResNet50 (AUC={auc_r:.4f})', linewidth=2)
    ax.plot(fpr_m, tpr_m, 'r-', label=f'MoE (AUC={auc_m:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC 曲线 (Micro-average)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_roc_curves.png'), dpi=150)
    plt.close()
    print("[图表] ROC曲线 → 05_roc_curves.png")

    # ── 图6：推理速度对比 ──
    fig, ax = plt.subplots(figsize=(6, 5))
    models_names = ['ResNet50\n(后融合)', 'MoE\n(门控融合)']
    times = [resnet_metrics['inference_time'], moe_metrics['inference_time']]
    colors = ['#4CAF50', '#FF9800']
    bars = ax.bar(models_names, times, color=colors, edgecolor='white', width=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{t*1000:.1f} ms', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('推理时间 (秒/对)')
    ax.set_title('单对眼底图推理速度对比')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_inference_speed.png'), dpi=150)
    plt.close()
    print("[图表] 推理速度 → 06_inference_speed.png")


def generate_report(resnet_metrics, moe_metrics, dataset_size=6392):
    """生成对比报告文本"""
    report_path = os.path.join(OUTPUT_DIR, 'comparison_report.md')
    lines = []

    lines.append("# ResNet50 vs MoE 对比实验报告\n")

    lines.append("## 1. 实验设置\n")
    lines.append(f"- 数据集: ODIR-5K，{dataset_size} 个患者（{'子集抽样' if USE_SUBSET else '全量'}），训练/验证 8:2")
    lines.append(f"- 类别: {', '.join(CLASS_NAMES)}")
    lines.append(f"- 设备: {DEVICE}")
    lines.append(f"- Epochs: {EPOCHS}")
    lines.append(f"- 阈值: {THRESHOLD}\n")

    lines.append("## 2. 方法说明\n")
    lines.append("### ResNet50（后融合）")
    lines.append("- 左眼/右眼分别通过同一个 ResNet50 推理")
    lines.append("- 两眼的 Sigmoid 概率取 max 作为最终结果")
    lines.append("- 优点: 模型简单、参数量少；缺点: 信息利用不充分\n")
    lines.append("### MoE 双流门控融合")
    lines.append("- 左右眼各通过共享 ResNet 提取特征（冻结）")
    lines.append("- 拼接特征后经门控网络分配权重给 8 个专家子网络")
    lines.append("- 专家投票得出最终结果")
    lines.append("- 优点: 特征级融合、自学习权重；缺点: 参数量大\n")

    lines.append("## 3. 总体指标对比\n")
    lines.append("| 指标 | ResNet50 | MoE | 胜出 |")
    lines.append("|------|----------|-----|------|")

    compare_keys = [
        ('macro_f1', 'Macro F1', True),
        ('micro_f1', 'Micro F1', True),
        ('macro_precision', 'Macro Precision', True),
        ('macro_recall', 'Macro Recall', True),
        ('macro_auc', 'Macro AUC', True),
    ]
    for key, name, higher_better in compare_keys:
        r_val = resnet_metrics.get(key, 0)
        m_val = moe_metrics.get(key, 0)
        if higher_better:
            winner = 'ResNet50' if r_val > m_val else 'MoE' if m_val > r_val else '平手'
        else:
            winner = 'ResNet50' if r_val < m_val else 'MoE' if m_val < r_val else '平手'
        lines.append(f"| {name} | {r_val:.4f} | {m_val:.4f} | {winner} |")

    lines.append("\n## 4. 逐类 F1 对比\n")
    lines.append("| 类别 | ResNet50 F1 | MoE F1 | 胜出 |")
    lines.append("|------|-------------|--------|------|")
    for name in CLASS_NAMES:
        rf = resnet_metrics.get(f'f1_{name}', 0)
        mf = moe_metrics.get(f'f1_{name}', 0)
        winner = 'ResNet50' if rf > mf else 'MoE' if mf > rf else '平手'
        lines.append(f"| {name} | {rf:.4f} | {mf:.4f} | {winner} |")

    lines.append(f"\n## 5. 推理速度\n")
    lines.append(f"- ResNet50: {resnet_metrics['inference_time']*1000:.1f} ms/对")
    lines.append(f"- MoE: {moe_metrics['inference_time']*1000:.1f} ms/对")

    lines.append(f"\n## 6. 结论\n")
    # 自动分析胜负
    r_win = 0
    m_win = 0
    for key, _, _ in compare_keys:
        r_val = resnet_metrics.get(key, 0)
        m_val = moe_metrics.get(key, 0)
        if r_val > m_val:
            r_win += 1
        elif m_val > r_val:
            m_win += 1

    if m_win > r_win:
        lines.append(f"MoE 双流门控融合在 {m_win}/{len(compare_keys)} 个指标上优于 ResNet50 后融合方案，"
                     f"说明**特征级门控融合能更有效地利用双眼信息**。")
    elif r_win > m_win:
        lines.append(f"ResNet50 后融合在 {r_win}/{len(compare_keys)} 个指标上优于 MoE，"
                     f"在小数据集下，**简单方案往往泛化能力更强**。")
    else:
        lines.append("两种方法各有千秋，可根据场景选择。")

    report = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[报告] 对比报告 → {report_path}")
    print(report)


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("眼底疾病分类 — ResNet50 vs MoE 对比实验")
    print("=" * 60)

    # ── 1. 加载数据 ──
    print("\n[1/5] 加载数据...")
    full_df = pd.read_csv(CSV_PATH)
    if USE_SUBSET:
        full_df = full_df.sample(n=SUBSET_SIZE, random_state=SEED).reset_index(drop=True)
        print(f"  [子集模式] 随机抽取 {SUBSET_SIZE} 样本")
    print(f"  总样本数: {len(full_df)}")

    # 划分训练/测试 (80/20, 按患者划分)
    n_total = len(full_df)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    indices = list(range(n_total))
    random.shuffle(indices)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_df = full_df.iloc[train_indices].reset_index(drop=True)
    test_df = full_df.iloc[test_indices].reset_index(drop=True)

    # 训练集中再分出 20% 作为验证
    n_train2 = int(0.8 * len(train_df))
    val_df = train_df.iloc[n_train2:].reset_index(drop=True)
    train_df = train_df.iloc[:n_train2].reset_index(drop=True)

    print(f"  训练集: {len(train_df)} | 验证集: {len(val_df)} | 测试集: {len(test_df)}")

    # ── 2. 创建 DataLoader ──
    # ResNet50 用单图数据集
    train_single = SingleEyeDataset(train_df, IMG_DIR, augment=True)
    val_single = SingleEyeDataset(val_df, IMG_DIR, augment=False)

    # MoE 用配对数据集
    train_paired = PairedEyeDataset(train_df, IMG_DIR, augment=True)
    val_paired = PairedEyeDataset(val_df, IMG_DIR, augment=False)
    test_paired = PairedEyeDataset(test_df, IMG_DIR, augment=False)

    resnet_train_loader = DataLoader(train_single, batch_size=BATCH_SIZE,
                                     shuffle=True, num_workers=NUM_WORKERS)
    resnet_val_loader = DataLoader(val_single, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS)

    moe_train_loader = DataLoader(train_paired, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
    moe_val_loader = DataLoader(val_paired, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS)

    test_loader = DataLoader(test_paired, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    # ── 3. 训练/加载 ResNet50 ──
    resnet_ckpt = os.path.join(OUTPUT_DIR, 'best_resnet50.pth')
    if os.path.exists(resnet_ckpt):
        print("\n[2/5] 发现 ResNet50 检查点，跳过训练直接加载...")
        resnet_model = ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        resnet_model.load_state_dict(torch.load(resnet_ckpt, map_location=DEVICE))
        resnet_history = {'train_loss': [], 'val_loss': [], 'val_f1': []}  # 占位
    else:
        print("\n[2/5] 训练 ResNet50...")
        resnet_model, resnet_history = train_resnet50(resnet_train_loader, resnet_val_loader)

    # ── 4. 训练 MoE ──
    print("\n[3/5] 训练 MoE 双流模型...")
    moe_model, moe_history = train_moe(moe_train_loader, moe_val_loader, epochs=6)

    # ── 4. 统一评估 ──
    print("\n[4/5] 统一评估两个模型...")
    print("  评估 ResNet50 (后融合)...")
    r_labels, r_probs, r_time = evaluate_paired(resnet_model, test_loader, method='resnet')
    r_metrics, r_preds = compute_all_metrics(r_labels, r_probs, THRESHOLD)
    r_metrics['inference_time'] = r_time

    print("  评估 MoE (门控融合)...")
    m_labels, m_probs, m_time = evaluate_paired(moe_model, test_loader, method='moe')
    m_metrics, m_preds = compute_all_metrics(m_labels, m_probs, THRESHOLD)
    m_metrics['inference_time'] = m_time

    # ── 5. 对比分析 ──
    print("\n[5/5] 生成对比图表和报告...")
    plot_comparison(r_metrics, m_metrics, r_probs, m_probs,
                    r_labels, m_labels, resnet_history, moe_history)
    generate_report(r_metrics, m_metrics, len(full_df))

    # ── 终端摘要 ──
    print("\n" + "=" * 60)
    print("对比实验完成！")
    print(f"  ResNet50 → Macro F1: {r_metrics['macro_f1']:.4f} | "
          f"AUC: {r_metrics['macro_auc']:.4f} | "
          f"推理: {r_metrics['inference_time']*1000:.1f}ms")
    print(f"  MoE      → Macro F1: {m_metrics['macro_f1']:.4f} | "
          f"AUC: {m_metrics['macro_auc']:.4f} | "
          f"推理: {m_metrics['inference_time']*1000:.1f}ms")
    print(f"  所有结果保存在: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
