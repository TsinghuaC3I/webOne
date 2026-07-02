import json

from PIL import Image
import os

def compress_image(input_path, output_path, target_size=(1024, 768)):
    """
    将图像压缩到指定尺寸

    参数:
    input_path: 输入图像路径
    output_path: 输出图像路径
    target_size: 目标尺寸，默认为(1024, 768)
    """
    try:
        # 打开原始图像
        with Image.open(input_path) as img:
            # 获取原始图像尺寸
            original_size = img.size
            # print(f"原始图像尺寸: {original_size}")

            # 调整图像尺寸
            resized_img = img.resize(target_size, Image.LANCZOS)

            # 保存调整后的图像
            resized_img.save(output_path)
            # print(f"图像已成功压缩到: {target_size}")
            # print(f"保存路径: {output_path}")

            return True

    except Exception as e:
        print(f"处理图像时出错: {e}")
        return False

def read_json(ffile):
    with open(ffile, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data
