import torch
import torch.nn as nn
import torchvision.models as models
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        """
        初始化 ResNet50 模型。

        参数：
        - num_classes: 分类任务的类别数量，默认为 2（适用于二分类问题）。
        - pretrained: 是否加载预训练权重，默认为 True。
        """
        super(ResNet50Model, self).__init__()

        self.resnet50 = models.resnet50(pretrained=pretrained)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入图像张量，形状为 (batch_size, 3, H, W)。

        返回：
        - logits: 输出的未归一化预测值，形状为 (batch_size, num_classes)。
        """
        return self.resnet50(x)

if __name__ == "__main__":
    dummy_input = torch.randn(4, 3, 224, 224)

    model = ResNet50Model(num_classes=2, pretrained=True)

    output = model(dummy_input)
    print("输出形状:", output.shape)