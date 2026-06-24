import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from mymodel import ResNet50Model

# 参数配置
model_path = "best_resnet50_test.pth"
img_path = "data/preprocessed_images/2346_right.jpg"
class_names = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7"]
threshold = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    model = ResNet50Model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# 单张图片推理
def predict_single_image(model, image_tensor, class_names, threshold):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs)
        predicted_labels = (probabilities > threshold).squeeze().cpu().numpy()
        confidence_scores = probabilities.squeeze().cpu().numpy()

    predicted_classes = [class_names[i] for i, label in enumerate(predicted_labels) if label == 1]
    return predicted_classes, confidence_scores


# 主推理逻辑
if __name__ == '__main__':
    model = load_model(model_path)

    image_tensor = preprocess_image(img_path)

    predicted_classes, confidence_scores = predict_single_image(model, image_tensor, class_names, threshold)

    print(f"Predicted Classes: {predicted_classes}")
    print("Confidence Scores:")
    for i, score in enumerate(confidence_scores):
        print(f"  {class_names[i]}: {score * 100:.2f}%")