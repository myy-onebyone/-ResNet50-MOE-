import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from mymodel import ResNet50Model
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 参数配置
model_path = "best_resnet50.pth"
excel_path = "data/random_sample_500_3.xlsx"
image_folder = "data/all_images"
class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]
threshold = 0.85
output_excel_path = "data/prediction_results_3.xlsx"

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
    return predicted_labels

# 主推理逻辑
if __name__ == '__main__':
    model = load_model(model_path)

    df = pd.read_excel(excel_path)

    all_true_labels = []
    all_pred_labels = []
    results = []  # 用于存储结果数据

    for index, row in df.iterrows():
        left_eye_image_name = row["Left-Fundus"]
        right_eye_image_name = row["Right-Fundus"]
        true_labels = row[class_names].values.astype(int)

        left_eye_image_path = os.path.join(image_folder, left_eye_image_name)
        if os.path.exists(left_eye_image_path):
            left_eye_tensor = preprocess_image(left_eye_image_path)
            left_eye_pred = predict_single_image(model, left_eye_tensor, class_names, threshold)
        else:
            print(f"Image not found: {left_eye_image_path}")
            left_eye_pred = [0] * len(class_names)

        right_eye_image_path = os.path.join(image_folder, right_eye_image_name)
        if os.path.exists(right_eye_image_path):
            right_eye_tensor = preprocess_image(right_eye_image_path)
            right_eye_pred = predict_single_image(model, right_eye_tensor, class_names, threshold)
        else:
            print(f"Image not found: {right_eye_image_path}")
            right_eye_pred = [0] * len(class_names)

        final_pred = [max(l, r) for l, r in zip(left_eye_pred, right_eye_pred)]

        if any(final_pred[1:]):
            final_pred[0] = 0

        if all(label == 0 for label in final_pred):
            final_pred[1] = 1

        # 判断预测是否完全正确
        is_correct = int(np.array_equal(true_labels, final_pred))  # 转换为 0 或 1

        # 存储结果
        result_row = {
            "Left-Eye-Image": left_eye_image_name,
            "Right-Eye-Image": right_eye_image_name,
            **{class_names[i]: int(final_pred[i]) for i in range(len(class_names))},  # 显式转换为整数
            "Is-Correct": is_correct  # 预测是否完全正确，也是 0 或 1
        }
        results.append(result_row)

        all_true_labels.append(true_labels)
        all_pred_labels.append(final_pred)

        print(f"Sample {index + 1}:")
        print(f"True Labels: {[class_names[i] for i, label in enumerate(true_labels) if label == 1]}")
        print(f"Predicted Labels: {[class_names[i] for i, label in enumerate(final_pred) if label == 1]}")
        print("-" * 50)

    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    precision = precision_score(all_true_labels, all_pred_labels, average='samples')
    recall = recall_score(all_true_labels, all_pred_labels, average='samples')
    f1 = f1_score(all_true_labels, all_pred_labels, average='samples')

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 将结果保存到 Excel 文件
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_excel_path, index=False)
    print(f"Results saved to {output_excel_path}")