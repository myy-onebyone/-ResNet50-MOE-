import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义数据集类
class dataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_excel(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        left_img_name = os.path.join(self.img_dir, row['Left-Fundus'])
        left_image = Image.open(left_img_name).convert('RGB')

        right_img_name = os.path.join(self.img_dir, row['Right-Fundus'])
        right_image = Image.open(right_img_name).convert('RGB')

        label_name = torch.tensor(row[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values.astype(float),
                                  dtype=torch.float32)

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return (left_image, right_image), label_name


# 定义共享特征提取器
class SharedFeatureExtractor(nn.Module):
    def __init__(self):
        super(SharedFeatureExtractor, self).__init__()
        # 加载预训练的 ResNet-50
        resnet = models.resnet50(pretrained=True)
        # 冻结 ResNet-50 的参数（可选）
        for param in resnet.parameters():
            param.requires_grad = False
        # 移除最后的全连接层，只保留卷积部分
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x_left, x_right):
        # 分别提取左眼和右眼的特征
        left_features = self.features(x_left).view(x_left.size(0), -1)  # 展平为 (batch_size, 2048)
        right_features = self.features(x_right).view(x_right.size(0), -1)  # 展平为 (batch_size, 2048)

        # 拼接左右眼特征
        combined_features = torch.cat((left_features, right_features), dim=1)  # (batch_size, 4096)
        return combined_features


# 定义专家子网络
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # 输出概率值


# 定义门控网络
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


# 定义门控专家网络
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, num_tasks, input_dim):
        super(MixtureOfExperts, self).__init__()
        self.shared_feature_extractor = SharedFeatureExtractor()
        self.experts = nn.ModuleList([Expert(input_dim, num_tasks) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x_left, x_right):
        shared_features = self.shared_feature_extractor(x_left, x_right)
        gate_weights = self.gating_network(shared_features)
        expert_outputs = [expert(shared_features) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        final_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return final_output


# 定义 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 防止 log(0)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# 数据路径
img_dir = "data/Training_Dataset"
train_excel_path = "data/train_set.xlsx"
val_excel_path = "data/val_set.xlsx"
test_excel_path = "data/test_set.xlsx"

# 数据增强与标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
train_dataset = dataset(img_dir=img_dir, csv_path=train_excel_path, transform=transform)
val_dataset = dataset(img_dir=img_dir, csv_path=val_excel_path, transform=transform)
test_dataset = dataset(img_dir=img_dir, csv_path=test_excel_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型、损失函数和优化器
num_experts = 8
num_tasks = 8
input_dim = 4096
model = MixtureOfExperts(num_experts, num_tasks, input_dim)
model.to(device)
criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)


# 训练和验证循环
def train_and_validate(model, train_loader, val_loader, num_epochs):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
        for (left_images, right_images), labels in train_bar:
            left_images, right_images, labels = left_images.to(device), right_images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(left_images, right_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for (left_images, right_images), labels in val_bar:
                left_images, right_images, labels = left_images.to(device), right_images.to(device), labels.to(device)

                outputs = model(left_images, right_images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved!")


# 测试阶段
def test_model(model, test_loader):
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    model.eval()
    total_test_loss = 0
    test_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for (left_images, right_images), labels in test_bar:
            left_images, right_images, labels = left_images.to(device), right_images.to(device), labels.to(device)

            outputs = model(left_images, right_images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            test_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")


# 开始训练和验证
train_and_validate(model, train_loader, val_loader, num_epochs=20)

# 测试模型
test_model(model, test_loader)