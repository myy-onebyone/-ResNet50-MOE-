import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, hamming_loss
)
from mymodel import ResNet50Model
from dataloader import dataset


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

set_seed()

# 参数配置
img_dir = "data/all_images"      # 修改为你的图片目录路径
csv_path = "data/full_df.csv"
batch_size = 32
num_workers = 4
epochs = 20
learning_rate = 0.001
output_dir = "output_visualizations"

NUM_CLASSES = 8
CLASS_NAMES = ["正常", "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他疾病/异常"]

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_dataset = dataset(img_dir=img_dir, csv_path=csv_path, augment=False, balance=True)

train_size = int(0.7 * len(full_dataset))
val_size = (len(full_dataset) - train_size) // 2
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = ResNet50Model(num_classes=NUM_CLASSES, pretrained=True).to(device)
criterion = nn.BCEWithLogitsLoss()   # 多标签分类损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def compute_metrics(outputs, labels, threshold=0.5):
    """计算多标签分类指标"""
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    # Subset accuracy (exact match ratio)
    exact_match = (preds == labels).all(dim=1).float().mean().item()

    # Per-label accuracy (汉明准确率)
    correct_per_label = (preds == labels).float()
    per_label_acc = correct_per_label.mean(dim=0).cpu().numpy()

    # 转为 numpy 用于 sklearn 指标
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 宏平均 F1
    try:
        macro_f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)
    except Exception:
        macro_f1 = 0.0

    return exact_match, per_label_acc, macro_f1


# 验证函数
def validate(loader, model, criterion):
    model.eval()
    total_loss = 0.0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    exact_match, per_label_acc, macro_f1 = compute_metrics(all_outputs, all_labels)

    avg_loss = total_loss / len(loader)
    return avg_loss, exact_match, per_label_acc, macro_f1, all_outputs, all_labels


# 测试函数
def test(loader, model, criterion):
    model.eval()
    total_loss = 0.0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    exact_match, per_label_acc, macro_f1 = compute_metrics(all_outputs, all_labels)

    avg_loss = total_loss / len(loader)

    print(f"\n{'='*50}")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Subset Accuracy (exact match): {exact_match:.2%}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"\nPer-class Accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {per_label_acc[i]:.2%}")
    print(f"{'='*50}")

    # 保存各类别 F1 曲线图
    probs = torch.sigmoid(all_outputs).cpu().numpy()
    preds = (probs > 0.5).astype(float)
    labels_np = all_labels.cpu().numpy()

    plot_per_class_f1(CLASS_NAMES, labels_np, preds)

    return avg_loss, exact_match, macro_f1


def plot_per_class_f1(class_names, labels_np, preds_np):
    """绘制各类别 F1 柱状图"""
    per_class_f1 = f1_score(labels_np, preds_np, average=None, zero_division=0)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(class_names)), per_class_f1, color='steelblue')
    plt.xticks(range(len(class_names)), class_names, rotation=30, ha='right')
    plt.ylim(0, 1)
    plt.ylabel('F1-score')
    plt.title('Per-class F1 Score')
    for bar, val in zip(bars, per_class_f1):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_f1.png"))
    plt.close()


# 主训练逻辑
def main():
    best_val_f1 = 0.0

    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_train_outputs, all_train_labels = [], []

        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                all_train_outputs.append(outputs.detach())
                all_train_labels.append(labels)

                tepoch.set_postfix(loss=loss.item())

        # 计算训练指标
        all_train_outputs = torch.cat(all_train_outputs, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        _, _, train_f1 = compute_metrics(all_train_outputs, all_train_labels)

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Macro F1: {train_f1:.4f}")

        # 验证
        val_loss, val_exact, val_per_label, val_f1, _, _ = validate(val_loader, model, criterion)
        print(f"Val Loss: {val_loss:.4f}, Val Exact Match: {val_exact:.2%}, Val Macro F1: {val_f1:.4f}")

        # 保存最佳模型（以验证 F1 为准）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_resnet50.pth')
            print(f">>> 保存最佳模型! (Macro F1: {best_val_f1:.4f})")

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

    # 绘图
    def plot_metrics():
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue', marker='o')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red', marker='o')
        plt.title('Loss Curve (BCEWithLogitsLoss)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), train_f1s, label='Train Macro F1', color='blue', marker='o')
        plt.plot(range(1, epochs + 1), val_f1s, label='Validation Macro F1', color='red', marker='o')
        plt.title('F1 Score Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Macro F1')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, "f1_curve.png"))
        plt.close()

    plot_metrics()

    print("\nTraining completed. Evaluating on test set...")
    test_loss, test_exact, test_f1 = test(test_loader, model, criterion)
    print(f"\nFinal Test Results: Loss={test_loss:.4f}, Exact Match={test_exact:.2%}, Macro F1={test_f1:.4f}")



if __name__ == '__main__':
    main()