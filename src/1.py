import os
from PIL import Image
import pandas as pd

def images_are_identical(image_path_1, image_path_2):
    """
    检查两张图片是否完全相同（逐像素比对）。
    :param image_path_1: 第一张图片的路径
    :param image_path_2: 第二张图片的路径
    :return: 如果完全相同返回 True，否则返回 False
    """
    try:
        # 打开两张图片
        img1 = Image.open(image_path_1)
        img2 = Image.open(image_path_2)

        # 检查尺寸和模式是否相同
        if img1.size != img2.size or img1.mode != img2.mode:
            return False

        # 逐像素比较
        pixels1 = list(img1.getdata())
        pixels2 = list(img2.getdata())

        return pixels1 == pixels2
    except Exception as e:
        print(f"无法处理图片 {image_path_1} 或 {image_path_2}: {e}")
        return False

def batch_compare_folders(folder1, folder2, output_excel):
    """
    批量比较两个文件夹中的图片，并将结果保存到 Excel 文件中。
    :param folder1: 第一个文件夹路径
    :param folder2: 第二个文件夹路径
    :param output_excel: 输出 Excel 文件路径
    """
    # 存储比对结果
    results = []
    # 获取两个文件夹中的所有 _left.jpg 图片
    files1 = [f for f in os.listdir(folder1) if f.endswith("_left.jpg")]
    files2 = [f for f in os.listdir(folder2) if f.endswith("_left.jpg")]

    # 初始化第二个文件夹的指针
    j = 400

    # 遍历第一个文件夹的所有图片
    for file_name1 in files1:
        file_path1 = os.path.join(folder1, file_name1)

        try:
            # 确保对应的 _right.jpg 存在
            file_name1_right = file_name1.replace("_left.jpg", "_right.jpg")
            file_path1_right = os.path.join(folder1, file_name1_right)
            if not os.path.exists(file_path1_right):
                continue

            # 从第二个文件夹的当前指针位置开始查找匹配的图片
            match_found = False
            while j < len(files2):
                file_name2 = files2[j]
                file_path2 = os.path.join(folder2, file_name2)

                try:
                    # 确保对应的 _right.jpg 存在
                    file_name2_right = file_name2.replace("_left.jpg", "_right.jpg")
                    file_path2_right = os.path.join(folder2, file_name2_right)
                    if not os.path.exists(file_path2_right):
                        j += 1
                        continue

                    # 比较两张图片是否完全相同
                    if images_are_identical(file_path1, file_path2):
                        results.append({
                            "Folder1_Image_Left": file_name1,
                            "Folder2_Image_Left": file_name2,
                            "Folder1_Image_Right": file_name1_right,
                            "Folder2_Image_Right": file_name2_right
                        })
                        print(f"匹配成功: {file_name1} -> {file_name2}")
                        print(f"自动推断: {file_name1_right} -> {file_name2_right}")
                        match_found = True
                        j += 1  # 移动第二个文件夹的指针
                        break
                    else:
                        # 如果不匹配，继续下一个图片
                        j += 1
                except Exception as e:
                    print(f"无法处理文件 {file_path2}: {e}")
                    j += 1

            if not match_found:
                # 如果没有找到匹配的图片，继续处理下一个图片
                print(f"未找到匹配: {file_name1}")
        except Exception as e:
            print(f"无法处理文件 {file_path1}: {e}")

    # 将结果保存到 Excel 文件
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"比对结果已保存到 {output_excel}")

# 示例用法
if __name__ == "__main__":
    # 替换为您的文件夹路径
    folder1 = "D:/【A07】基于眼底医学影像的眼科疾病智能诊断系统【诚迈科技】训练集及标注/4.3【A07】基于眼底医学影像的眼科疾病智能诊断系统【诚迈科技】验证集(1)/4.3【A07】基于眼底医学影像的眼科疾病智能诊断系统【诚迈科技】验证集(1)/验证集/验证集图像"
    folder2 = "D:/【A07】基于眼底医学影像的眼科疾病智能诊断系统【诚迈科技】训练集及标注/训练集及标注/Training Images/Training Images"

    # 输出 Excel 文件路径
    output_excel = "results_no_sort.xlsx"

    # 检查路径是否存在
    if not os.path.isdir(folder1):
        print(f"第一个文件夹路径无效: {folder1}")
    elif not os.path.isdir(folder2):
        print(f"第二个文件夹路径无效: {folder2}")
    else:
        batch_compare_folders(folder1, folder2, output_excel)