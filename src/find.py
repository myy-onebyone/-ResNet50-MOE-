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

    # 遍历第一个文件夹中的所有图片
    for root1, _, files1 in os.walk(folder1):
        for file_name1 in files1:
            # 只处理 _left.jpg 的图片
            if not file_name1.endswith("_left.jpg"):
                continue

            file_path1 = os.path.join(root1, file_name1)
            try:
                # 检查是否为支持的图片格式
                if not file_path1.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue

                print(f"正在处理图片: {file_name1}")

                # 获取对应的 _right.jpg 图片名
                file_name1_right = file_name1.replace("_left.jpg", "_right.jpg")
                file_path1_right = os.path.join(root1, file_name1_right)

                # 遍历第二个文件夹中的所有图片
                match_found = False
                for root2, _, files2 in os.walk(folder2):
                    for file_name2 in files2:
                        # 只处理 _left.jpg 的图片
                        if not file_name2.endswith("_left.jpg"):
                            continue

                        file_path2 = os.path.join(root2, file_name2)
                        try:
                            # 检查是否为支持的图片格式
                            if not file_path2.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                                continue

                            # 比较两张图片是否完全相同
                            if images_are_identical(file_path1, file_path2):
                                # 获取对应的 _right.jpg 图片名
                                file_name2_right = file_name2.replace("_left.jpg", "_right.jpg")
                                file_path2_right = os.path.join(root2, file_name2_right)

                                # 确保对应的 _right.jpg 存在
                                if os.path.exists(file_path1_right) and os.path.exists(file_path2_right):
                                    results.append({
                                        "Folder1_Image_Left": file_name1,
                                        "Folder2_Image_Left": file_name2,
                                        "Folder1_Image_Right": file_name1_right,
                                        "Folder2_Image_Right": file_name2_right
                                    })
                                    print(f"匹配成功: {file_name1} -> {file_name2}")
                                    print(f"自动推断: {file_name1_right} -> {file_name2_right}")
                                    match_found = True
                                    break  # 匹配成功后跳出内层循环
                        except Exception as e:
                            print(f"无法处理文件 {file_path2}: {e}")
                    if match_found:
                        break  # 匹配成功后跳出外层循环
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
    output_excel = "results.xlsx"

    # 检查路径是否存在
    if not os.path.isdir(folder1):
        print(f"第一个文件夹路径无效: {folder1}")
    elif not os.path.isdir(folder2):
        print(f"第二个文件夹路径无效: {folder2}")
    else:
        batch_compare_folders(folder1, folder2, output_excel)