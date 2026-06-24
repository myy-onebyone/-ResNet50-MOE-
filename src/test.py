import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mymodel import ResNet50Model
from dataloader import dataset

# 设置随机种子以便结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
set_seed()

# 参数配置
img_dir = "data/preprocessed_images"
csv_path = "data/full_df.csv"
batch_size = 64
num_workers = 4
output_dir = "output_visualizations"

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_dataset = dataset(img_dir=img_dir, csv_path=csv_path, augment=False, balance=False)

train_size = int(0.7 * len(full_dataset))
val_size = (len(full_dataset) - train_size) // 2
test_size = len(full_dataset) - train_size - val_size

_, _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = ResNet50Model(num_classes=8, pretrained=False).to(device)
model.load_state_dict(torch.load('best_resnet50.pth', map_location=device))
model.eval()

criterion = torch.nn.CrossEntropyLoss()

# 测试函数
def test(loader, model, criterion):
    model.eval()
    all_labels = []
    all_preds = []

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    plot_confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy

# 绘制混淆矩阵
def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(8))  # 假设有 8 个类别
    disp.plot(cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

if __name__ == '__main__':
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = test(test_loader, model, criterion)