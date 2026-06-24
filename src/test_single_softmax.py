import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from mymodel import ResNet50Model

# 参数配置
model_path = "best_resnet50_test.pth"
img_path = "data/preprocessed_images/832_left.jpg"
class_names = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7"]

# 检查是否存在GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义图像预处理操作
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)


# 初始化模型并加载权重
def load_model(model_path):
    model = ResNet50Model(num_classes=8, pretrained=False).to(device)  # 改为 8 分类
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型权重
    model.eval()
    return model


# 单张图片推理
def predict_single_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted_idx].item()
    return predicted_class, confidence


# 主推理逻辑
if __name__ == '__main__':
    model = load_model(model_path)

    image_tensor = preprocess_image(img_path)

    predicted_class, confidence = predict_single_image(model, image_tensor, class_names)

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")