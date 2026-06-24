import pandas as pd
import os
import shutil

# 配置参数
excel_file = 'data/test.xlsx'
image_folder = 'data/Training Images/Training Images'
output_folder = 'data/test_images/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df = pd.read_excel(excel_file)

column1 = 'Left-Fundus'
column2 = 'Right-Fundus'

# 去重
image_names = set(df[column1].dropna()).union(set(df[column2].dropna()))

for image_name in image_names:
    source_path = os.path.join(image_folder, image_name)
    destination_path = os.path.join(output_folder, image_name)

    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"已复制: {image_name}")
    else:
        print(f"文件未找到: {image_name}")

print("处理完成")