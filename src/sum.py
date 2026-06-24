import pandas as pd


# 读取 Excel 文件并统计大标签出现次数
def count_combined_labels(file_path, label_columns):
    """
    统计每个大标签（由多个小标签组成）的出现次数。

    参数:
        file_path (str): Excel 文件路径。
        label_columns (list): 标签列的名称列表。

    返回:
        pd.Series: 每个大标签的出现次数。
    """
    df = pd.read_excel(file_path)

    for col in label_columns:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不在表格中，请检查列名是否正确。")

    df['Combined_Label'] = df[label_columns].astype(str).apply(''.join, axis=1)

    combined_label_counts = df['Combined_Label'].value_counts()

    return combined_label_counts


# 主函数
if __name__ == "__main__":
    file_path = "data/Traning_Dataset.xlsx"

    label_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

    try:
        counts = count_combined_labels(file_path, label_columns)

        print("大标签出现次数：")
        print(counts)

        counts.to_excel("combined_label_counts.xlsx", header=["Count"])
        print("\n统计结果已保存到 'combined_label_counts.xlsx'")
    except Exception as e:
        print(f"发生错误：{e}")