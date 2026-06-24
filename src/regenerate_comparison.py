"""
眼底疾病多标签分类 — 对比实验图表重新生成
ResNet50 (后融合) vs MoE 双流门控融合

基于已训练好的模型，重新评估 + 生成高质量对比图表。
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
CLASS_NAMES_EN = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Other"]

BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 6
LR = 0.001
THRESHOLD = 0.5
SEED = 42
USE_SUBSET = True
SUBSET_SIZE = 800

# 颜色方案
COLOR_RESNET = '#2196F3'      # 蓝色
COLOR_MOE = '#FF5722'         # 深橙
COLOR_RESNET_LIGHT = '#BBDEFB'
COLOR_MOE_LIGHT = '#FFCCBC'

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
        left_img = Image.open(os.path.join(self.img_dir, row['Left-Fundus'])).convert('RGB')
        right_img = Image.open(os.path.join(self.img_dir, row['Right-Fundus'])).convert('RGB')
        left_tensor = self.transform(left_img)
        right_tensor = self.transform(right_img)
        target = torch.tensor(ast.literal_eval(row['target']), dtype=torch.float32)
        return (left_tensor, right_tensor), target


class SingleEyeDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.img_dir = img_dir
        self.augment = augment
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
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        for param in resnet.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)


class Expert(nn.Module):
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
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_experts), nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts=8, num_tasks=8, input_dim=4096):
        super().__init__()
        self.shared_extractor = SharedFeatureExtractor()
        self.experts = nn.ModuleList([Expert(input_dim, num_tasks) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, x_left, x_right):
        left_feat = self.shared_extractor(x_left)
        right_feat = self.shared_extractor(x_right)
        combined = torch.cat([left_feat, right_feat], dim=1)
        gate_weights = self.gating(combined)
        expert_outs = torch.stack([expert(combined) for expert in self.experts], dim=1)
        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outs, dim=1)
        return output


# ==================== FocalLoss ====================
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
    for i, name in enumerate(CLASS_NAMES):
        metrics[f'f1_{name}'] = per_class_f1[i]
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
    model = MixtureOfExperts(num_experts=8, num_tasks=NUM_CLASSES, input_dim=4096).to(DEVICE)
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


# ==================== 统一评估 ====================
def evaluate_paired(model, paired_loader, method='resnet'):
    model.eval()
    all_probs, all_labels = [], []
    inference_times = []

    with torch.no_grad():
        for (left, right), labels in tqdm(paired_loader, desc=f"评估 {method}"):
            left, right = left.to(DEVICE), right.to(DEVICE)
            t0 = time.time()
            if method == 'resnet':
                left_out = torch.sigmoid(model(left))
                right_out = torch.sigmoid(model(right))
                probs = torch.max(left_out, right_out)
            else:
                outputs = model(left, right)
                probs = torch.sigmoid(outputs)
            t1 = time.time()
            inference_times.append(t1 - t0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    avg_time = np.mean(inference_times)

    return all_labels, all_probs, avg_time


# ==================== 可视化（改进版） ====================
def plot_comparison(resnet_metrics, moe_metrics, resnet_probs, moe_probs,
                    resnet_labels, moe_labels, resnet_history, moe_history):
    """生成6幅高质量对比图表"""

    # ---- 中文字体设置 ----
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    sns.set_style("whitegrid")

    # ============================================================
    # 图1：训练曲线对比（双Y轴 + 清晰图例）
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    epochs_range = range(1, len(resnet_history['train_loss']) + 1)
    epochs_moe = range(1, len(moe_history['train_loss']) + 1)

    # Loss曲线
    ax = axes[0]
    ax.plot(epochs_range, resnet_history['train_loss'], '-', color=COLOR_RESNET,
            linewidth=2, alpha=0.5, marker='o', markersize=4, label='ResNet50 训练')
    ax.plot(epochs_range, resnet_history['val_loss'], '-', color=COLOR_RESNET,
            linewidth=2.5, marker='s', markersize=5, label='ResNet50 验证')
    ax.plot(epochs_moe, moe_history['train_loss'], '-', color=COLOR_MOE,
            linewidth=2, alpha=0.5, marker='o', markersize=4, label='MoE 训练')
    ax.plot(epochs_moe, moe_history['val_loss'], '-', color=COLOR_MOE,
            linewidth=2.5, marker='s', markersize=5, label='MoE 验证')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss 曲线对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # F1曲线
    ax = axes[1]
    ax.plot(epochs_range, resnet_history['val_f1'], '-', color=COLOR_RESNET,
            linewidth=2.5, marker='o', markersize=6, label='ResNet50')
    ax.plot(epochs_moe, moe_history['val_f1'], '-', color=COLOR_MOE,
            linewidth=2.5, marker='s', markersize=6, label='MoE')
    # 标注最优值
    best_rn_f1 = max(resnet_history['val_f1'])
    best_rn_epoch = resnet_history['val_f1'].index(best_rn_f1) + 1
    best_moe_f1 = max(moe_history['val_f1'])
    best_moe_epoch = moe_history['val_f1'].index(best_moe_f1) + 1
    ax.annotate(f'{best_rn_f1:.4f}', xy=(best_rn_epoch, best_rn_f1),
                xytext=(best_rn_epoch-0.5, best_rn_f1+0.02),
                fontsize=9, color=COLOR_RESNET, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_RESNET, alpha=0.5))
    ax.annotate(f'{best_moe_f1:.4f}', xy=(best_moe_epoch, best_moe_f1),
                xytext=(best_moe_epoch-0.5, best_moe_f1-0.04),
                fontsize=9, color=COLOR_MOE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_MOE, alpha=0.5))
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Macro F1', fontsize=12)
    ax.set_title('验证集 Macro F1 曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(OUTPUT_DIR, '01_training_curves.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[图表1/6] 训练曲线 → 01_training_curves.png")

    # ============================================================
    # 图2：总体指标对比（分组柱状图 + 数值标注 + 差异箭头）
    # ============================================================
    metric_names = ['macro_f1', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_auc']
    metric_labels = ['Macro F1', 'Micro F1', 'Macro Precision', 'Macro Recall', 'Macro AUC']

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(metric_names))
    width = 0.32

    resnet_vals = [resnet_metrics.get(m, 0) for m in metric_names]
    moe_vals = [moe_metrics.get(m, 0) for m in metric_names]
    diffs = [m - r for r, m in zip(resnet_vals, moe_vals)]

    bars1 = ax.bar(x - width/2, resnet_vals, width, label='ResNet50 (后融合)',
                   color=COLOR_RESNET, edgecolor='white', linewidth=0.8, alpha=0.9)
    bars2 = ax.bar(x + width/2, moe_vals, width, label='MoE (门控融合)',
                   color=COLOR_MOE, edgecolor='white', linewidth=0.8, alpha=0.9)

    # 数值标注
    for bar, val in zip(bars1, resnet_vals):
        color = '#333' if val > 0.15 else '#999'
        offset = 0.015
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f'{val:.4f}', ha='center', fontsize=9, fontweight='bold', color=color)
    for bar, val in zip(bars2, moe_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{val:.4f}', ha='center', fontsize=9, fontweight='bold', color='#333')

    # 差异标注
    for i, diff in enumerate(diffs):
        mid_x = x[i]
        top_y = max(resnet_vals[i], moe_vals[i]) + 0.08
        sign = '+' if diff > 0 else ''
        ax.annotate(f'Δ={sign}{diff:.3f}', xy=(mid_x, top_y),
                    ha='center', fontsize=8, color='#D84315' if diff > 0 else '#1565C0',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9C4', alpha=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, max(max(resnet_vals), max(moe_vals)) * 1.35)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('ResNet50 vs MoE — 总体指标对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_overall_metrics.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[图表2/6] 总体指标 → 02_overall_metrics.png")

    # ============================================================
    # 图3：逐类 F1 对比（水平条形图 + 热力值标注）
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(CLASS_NAMES))
    height = 0.32

    resnet_f1s = [resnet_metrics.get(f'f1_{name}', 0) for name in CLASS_NAMES]
    moe_f1s = [moe_metrics.get(f'f1_{name}', 0) for name in CLASS_NAMES]

    bars1 = ax.barh(y_pos - height/2, resnet_f1s, height, label='ResNet50 (后融合)',
                    color=COLOR_RESNET, edgecolor='white', linewidth=0.8)
    bars2 = ax.barh(y_pos + height/2, moe_f1s, height, label='MoE (门控融合)',
                    color=COLOR_MOE, edgecolor='white', linewidth=0.8)

    # 数值标注
    for bar, val in zip(bars1, resnet_f1s):
        x_pos = max(bar.get_width() + 0.008, 0.005)
        color = '#333' if val > 0.05 else '#999'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=9, fontweight='bold', color=color)
    for bar, val in zip(bars2, moe_f1s):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=9, fontweight='bold', color='#333')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_xlim(0, max(max(resnet_f1s), max(moe_f1s)) * 1.35)
    ax.set_title('ResNet50 vs MoE — 逐类 F1 对比', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_per_class_f1.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[图表3/6] 逐类F1 → 03_per_class_f1.png")

    # ============================================================
    # 图4：逐类 AUC 对比（含数值热力图）
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(CLASS_NAMES))
    height = 0.32

    resnet_aucs = [resnet_metrics.get(f'auc_{name}', 0) for name in CLASS_NAMES]
    moe_aucs = [moe_metrics.get(f'auc_{name}', 0) for name in CLASS_NAMES]

    bars1 = ax.barh(y_pos - height/2, resnet_aucs, height, label='ResNet50 (后融合)',
                    color=COLOR_RESNET, edgecolor='white', linewidth=0.8)
    bars2 = ax.barh(y_pos + height/2, moe_aucs, height, label='MoE (门控融合)',
                    color=COLOR_MOE, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars1, resnet_aucs):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                va='center', fontsize=9, fontweight='bold', color='#333')
    for bar, val in zip(bars2, moe_aucs):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                va='center', fontsize=9, fontweight='bold', color='#333')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_xlim(0, max(max(resnet_aucs), max(moe_aucs)) * 1.3)
    ax.set_title('ResNet50 vs MoE — 逐类 AUC 对比', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_per_class_auc.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[图表4/6] 逐类AUC → 04_per_class_auc.png")

    # ============================================================
    # 图5：ROC 曲线（微平均 + 逐类子图）
    # ============================================================
    fig = plt.figure(figsize=(16, 12))

    # 子图1：微平均 ROC (大图)
    ax1 = plt.subplot(2, 3, (1, 3))  # 左侧大图占3列

    fpr_r, tpr_r, _ = roc_curve(resnet_labels.ravel(), resnet_probs.ravel())
    auc_r = auc(fpr_r, tpr_r)
    fpr_m, tpr_m, _ = roc_curve(moe_labels.ravel(), moe_probs.ravel())
    auc_m = auc(fpr_m, tpr_m)

    ax1.plot(fpr_r, tpr_r, '-', color=COLOR_RESNET, linewidth=2.5,
             label=f'ResNet50 (AUC={auc_r:.4f})')
    ax1.plot(fpr_m, tpr_m, '-', color=COLOR_MOE, linewidth=2.5,
             label=f'MoE (AUC={auc_m:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.4, linewidth=1)
    ax1.fill_between(fpr_r, tpr_r, alpha=0.05, color=COLOR_RESNET)
    ax1.fill_between(fpr_m, tpr_m, alpha=0.05, color=COLOR_MOE)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve (Micro-average)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # 子图2-6：逐类 ROC（选择5个有代表性的类别）
    # 按MoE AUC排序选top5
    class_aucs = [(name, moe_metrics.get(f'auc_{name}', 0)) for name in CLASS_NAMES]
    class_aucs.sort(key=lambda x: -x[1])
    top5_classes = class_aucs[:5]

    for idx, (cls_name, _) in enumerate(top5_classes):
        cls_idx = CLASS_NAMES.index(cls_name)
        ax = plt.subplot(2, 3, idx + 4)

        fpr_r_c, tpr_r_c, _ = roc_curve(resnet_labels[:, cls_idx], resnet_probs[:, cls_idx])
        auc_r_c = auc(fpr_r_c, tpr_r_c)
        fpr_m_c, tpr_m_c, _ = roc_curve(moe_labels[:, cls_idx], moe_probs[:, cls_idx])
        auc_m_c = auc(fpr_m_c, tpr_m_c)

        ax.plot(fpr_r_c, tpr_r_c, '-', color=COLOR_RESNET, linewidth=1.8,
                label=f'ResNet50 ({auc_r_c:.3f})')
        ax.plot(fpr_m_c, tpr_m_c, '-', color=COLOR_MOE, linewidth=1.8,
                label=f'MoE ({auc_m_c:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
        ax.set_title(f'{cls_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.2)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.suptitle('ResNet50 vs MoE — ROC 曲线分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(OUTPUT_DIR, '05_roc_curves.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[图表5/6] ROC曲线 → 05_roc_curves.png")

    # ============================================================
    # 图6：推理速度 + 模型参数量双面板对比
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    # 6a: 推理速度
    ax = axes[0]
    models_names = ['ResNet50\n(后融合)', 'MoE\n(门控融合)']
    times = [resnet_metrics['inference_time'] * 1000, moe_metrics['inference_time'] * 1000]
    colors_speed = [COLOR_RESNET, COLOR_MOE]
    bars = ax.bar(models_names, times, color=colors_speed, edgecolor='white', linewidth=1.5, width=0.55)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{t:.1f} ms', ha='center', fontsize=14, fontweight='bold', color='#333')
    ax.set_ylabel('推理时间 (ms/对)', fontsize=12)
    ax.set_title('单对眼底图推理速度', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(times) * 1.3)
    # 添加速度差异
    diff_ms = times[1] - times[0]
    speed_text = f'差异: {abs(diff_ms):.1f}ms ({"MoE慢" if diff_ms > 0 else "ResNet50慢"} {abs(diff_ms)/times[0]*100:.1f}%)'
    ax.text(0.5, max(times) * 1.1, speed_text, ha='center', fontsize=9,
            color='#666', transform=ax.get_xaxis_transform())

    # 6b: 模型参数量对比
    ax = axes[1]

    # 计算参数量
    resnet_model_tmp = ResNet50Classifier(num_classes=8, pretrained=False)
    moe_model_tmp = MixtureOfExperts(num_experts=8, num_tasks=8, input_dim=4096)

    rn_params = sum(p.numel() for p in resnet_model_tmp.parameters())
    me_params = sum(p.numel() for p in moe_model_tmp.parameters())
    # MoE中可训练参数
    me_trainable = sum(p.numel() for p in moe_model_tmp.parameters() if p.requires_grad)

    param_data = {
        'ResNet50\n(后融合)': rn_params,
        'MoE\n(总参数)': me_params,
        'MoE\n(可训练)': me_trainable,
    }
    param_colors = [COLOR_RESNET, COLOR_MOE, '#FFAB91']

    param_bars = ax.bar(param_data.keys(), param_data.values(),
                        color=param_colors, edgecolor='white', linewidth=1.5, width=0.55)
    for bar, val in zip(param_bars, param_data.values()):
        if val > 1e6:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5e6,
                    f'{val/1e6:.1f}M', ha='center', fontsize=12, fontweight='bold', color='#333')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5e6,
                    f'{val/1e3:.0f}K', ha='center', fontsize=12, fontweight='bold', color='#333')
    ax.set_ylabel('参数量', fontsize=12)
    ax.set_title('模型参数量对比', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(OUTPUT_DIR, '06_model_comparison.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[图表6/6] 推理速度+参数量 → 06_model_comparison.png")


def generate_report(resnet_metrics, moe_metrics, dataset_size=6392):
    """生成对比报告"""
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
    print("眼底疾病分类 — ResNet50 vs MoE 对比实验（图表重新生成）")
    print("=" * 60)

    # ── 1. 加载数据 ──
    print("\n[1/5] 加载数据...")
    full_df = pd.read_csv(CSV_PATH)
    if USE_SUBSET:
        full_df = full_df.sample(n=SUBSET_SIZE, random_state=SEED).reset_index(drop=True)
        print(f"  [子集模式] 随机抽取 {SUBSET_SIZE} 样本")
    print(f"  总样本数: {len(full_df)}")

    # 划分训练/测试 (80/20)
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
    train_single = SingleEyeDataset(train_df, IMG_DIR, augment=True)
    val_single = SingleEyeDataset(val_df, IMG_DIR, augment=False)

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
        print("\n[2/5] 发现 ResNet50 检查点，加载模型并重新获取训练历史...")
        # 由于原检查点可能无训练历史，重新训练以获取完整曲线
        print("  重新训练以获取完整训练曲线...")
        resnet_model, resnet_history = train_resnet50(resnet_train_loader, resnet_val_loader)
    else:
        print("\n[2/5] 训练 ResNet50...")
        resnet_model, resnet_history = train_resnet50(resnet_train_loader, resnet_val_loader)

    # ── 4. 训练 MoE ──
    moe_ckpt = os.path.join(OUTPUT_DIR, 'best_moe.pth')
    if os.path.exists(moe_ckpt):
        print("\n[3/5] 发现 MoE 检查点，重新训练以获取完整训练历史...")
        moe_model, moe_history = train_moe(moe_train_loader, moe_val_loader, epochs=EPOCHS)
    else:
        print("\n[3/5] 训练 MoE 双流模型...")
        moe_model, moe_history = train_moe(moe_train_loader, moe_val_loader, epochs=EPOCHS)

    # ── 5. 统一评估 ──
    print("\n[4/5] 统一评估两个模型...")
    print("  评估 ResNet50 (后融合)...")
    r_labels, r_probs, r_time = evaluate_paired(resnet_model, test_loader, method='resnet')
    r_metrics, r_preds = compute_all_metrics(r_labels, r_probs, THRESHOLD)
    r_metrics['inference_time'] = r_time

    print("  评估 MoE (门控融合)...")
    m_labels, m_probs, m_time = evaluate_paired(moe_model, test_loader, method='moe')
    m_metrics, m_preds = compute_all_metrics(m_labels, m_probs, THRESHOLD)
    m_metrics['inference_time'] = m_time

    # ── 6. 生成对比图表 ──
    print("\n[5/5] 生成对比图表和报告...")
    plot_comparison(r_metrics, m_metrics, r_probs, m_probs,
                    r_labels, m_labels, resnet_history, moe_history)
    generate_report(r_metrics, m_metrics, len(full_df))

    # ── 终端摘要 ──
    print("\n" + "=" * 60)
    print("对比实验图表重新生成完成！")
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
