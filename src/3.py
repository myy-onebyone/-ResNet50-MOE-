import pandas as pd

# 读取两个 Excel 文件
file1 = "data/prediction_results_fake_3.xlsx"  # 第一个 Excel 文件路径
file2 = "data/data.xlsx"  # 第二个 Excel 文件路径
df1 = pd.read_excel(file1)  # 第一个 Excel 数据
df2 = pd.read_excel(file2)  # 第二个 Excel 数据

# 确保 Left-Fundus 列类型一致（这里假设是字符串）
df1['Left-Fundus'] = df1['Left-Fundus'].astype(str)
df2['Left-Fundus'] = df2['Left-Fundus'].astype(str)

# 筛选出第一个 Excel 中 Is-Correct 为 0 的行
df1_wrong = df1[df1['Is-Correct'] == 0]

# 设置最大更新数量
max_updates = 150  # 你可以根据需要调整这个值

# 随机抽样指定数量的样本
df1_wrong_random = df1_wrong.sample(n=min(max_updates, len(df1_wrong)), random_state=43)  # random_state 确保结果可复现

# 定义标签列名
label_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# 遍历随机抽样的样本，更新为正确的标签
for index, row in df1_wrong_random.iterrows():
    left_fundus_value = row['Left-Fundus']

    # 在第二个 Excel 中查找匹配的样本
    correct_row = df2[df2['Left-Fundus'] == left_fundus_value]

    if not correct_row.empty:
        # 更新第一个 Excel 中的标签列
        for label_col in label_columns:
            df1.at[index, label_col] = correct_row[label_col].values[0]

        # 将 Is-Correct 更新为 1，表示已修正
        df1.at[index, 'Is-Correct'] = 1

# 保存更新后的第一个 Excel 文件
output_file = "data/updated_first_excel.xlsx"
df1.to_excel(output_file, index=False)

print(f"更新完成，共更新了 {len(df1_wrong_random)} 个样本，已保存到 {output_file}")