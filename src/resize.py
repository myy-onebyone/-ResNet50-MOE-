import os
import cv2


def crop_and_resize(image_path, output_path, target_size=(512, 512)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像：{image_path}，跳过该文件！")
        return

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 阈值分割，去除黑色背景
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print(f"未找到有效轮廓：{image_path}，跳过该文件！")
        return

    # 找到最大的轮廓
    cnt = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(cnt)

    cropped_image = image[y:y + h, x:x + w]

    resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, resized_image)
    print(f"已处理并保存：{output_path}")


def batch_process_images(input_folder, output_folder, target_size=(512, 512)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                crop_and_resize(input_path, output_path, target_size)
            except Exception as e:
                print(f"处理文件 {filename} 时出错：{e}")


input_folder = 'data/Training Images'
output_folder = 'data/all_images'
batch_process_images(input_folder, output_folder)