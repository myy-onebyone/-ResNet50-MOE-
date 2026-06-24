import pandas as pd

# 读取原始 Excel 文件
file_path = 'data/data.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 随机抽取 500 行
random_sample = df.sample(n=500, random_state=44)  # random_state 保证结果可重复
# 保存到新文件
output_path = 'data/random_sample_500_3.xlsx'
random_sample.to_excel(output_path, index=False)

print(f"已随机抽取 500 行并保存到 {output_path}")