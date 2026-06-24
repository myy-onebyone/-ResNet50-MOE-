import os
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
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
img_dir = "data/preprocessed_images"
csv_path = "data/train.csv"
batch_size = 16
num_workers = 4
epochs = 50
learning_rate = 0.001
output_dir = "output_visualizations"

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载全部数据集
full_dataset = dataset(img_dir=img_dir, csv_path=csv_path, augment=False, balance=True)

# 使用全部数据进行训练
train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = ResNet50Model(num_classes=8, pretrained=True).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 主训练逻辑
def main():
    best_train_accuracy = 0.0

    train_losses, train_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tepoch.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Average Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # 如果当前训练准确率更高，则保存模型
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            torch.save(model.state_dict(), 'best_resnet50.pth')
            print("保存最佳模型!")

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

    def plot_metrics(train_losses, train_accuracies):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue', marker='o')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', color='blue', marker='o')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
        plt.close()

    plot_metrics(train_losses, train_accuracies)

    print("\nTraining completed.")


if __name__ == '__main__':
    main()