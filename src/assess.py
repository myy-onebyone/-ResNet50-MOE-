import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# 参数配置
pred_excel_path = "data/prediction_results.xlsx"
true_excel_path = "data/data.xlsx"
sample_name_column = "Left-Fundus"
class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]


# 加载 Excel 数据
def load_data(excel_path, sample_name_column, class_names):
    df = pd.read_excel(excel_path)
    if sample_name_column not in df.columns:
        raise ValueError(f"Column '{sample_name_column}' not found in {excel_path}")
    for col in class_names:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {excel_path}")
    data = df.set_index(sample_name_column)[class_names]  # 设置样本名为索引
    return data


# 主计算逻辑
if __name__ == '__main__':
    # 加载预测标签和真实标签
    pred_labels_df = load_data(pred_excel_path, sample_name_column, class_names)
    true_labels_df = load_data(true_excel_path, sample_name_column, class_names)

    # 过滤真实标签，仅保留存在于预测标签中的样本
    common_samples = pred_labels_df.index.intersection(true_labels_df.index)

    if len(common_samples) == 0:
        raise ValueError("No common samples found between the two datasets.")

    # 对齐样本顺序
    pred_labels_aligned = pred_labels_df.loc[common_samples].values
    true_labels_aligned = true_labels_df.loc[common_samples].values

    # 计算评价指标
    accuracy = accuracy_score(true_labels_aligned, pred_labels_aligned)
    precision = precision_score(true_labels_aligned, pred_labels_aligned, average='samples')
    recall = recall_score(true_labels_aligned, pred_labels_aligned, average='samples')
    f1 = f1_score(true_labels_aligned, pred_labels_aligned, average='samples')

    # 输出结果
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 可选：输出被评估的样本数量
    print(f"\nNumber of evaluated samples: {len(common_samples)}")

    # 生成分类报告（按类别）
    report = classification_report(
        true_labels_aligned, pred_labels_aligned,
        target_names=class_names, output_dict=True
    )
    report_df = pd.DataFrame(report).T

    # 可视化：条形图展示每个类别的 Precision、Recall 和 F1-Score
    metrics_df = report_df.loc[class_names, ["precision", "recall", "f1-score"]]
    metrics_df.plot(kind="bar", figsize=(12, 6), colormap="viridis")
    plt.title("Per-Class Evaluation Metrics")
    plt.ylabel("Score")
    plt.xlabel("Class")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 可视化：混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(
        true_labels_aligned.argmax(axis=1),
        pred_labels_aligned.argmax(axis=1),
        labels=np.arange(len(class_names))
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    # 可视化：总体指标分布
    overall_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    overall_metrics_df = pd.Series(overall_metrics).to_frame(name="Value")
    overall_metrics_df.plot(kind="bar", legend=False, figsize=(8, 5), color="skyblue")
    plt.title("Overall Evaluation Metrics")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()