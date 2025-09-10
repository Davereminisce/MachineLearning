import os

# --- Path Initialization ---
# 训练集和测试集的基础路径
BASE_DIR = os.path.join("data", "catVSdog")
# 训练集路径
TRAIN_DIR = os.path.join(BASE_DIR, "train_data")
# 测试集路径
VALID_DIR = os.path.join(BASE_DIR, "test_data")
# 训练集 txt 文件路径
TRAIN_TXT_PATH = os.path.join(BASE_DIR, "train.txt")
# 测试集 txt 文件路径
VALID_TXT_PATH = os.path.join(BASE_DIR, "test.txt")

# Define the classes and their corresponding labels
# 定义类别和它们的标签
# This can make the code more flexible if you add more classes
# 这样在新增类别时，代码会更灵活
CLASSES = {
    'cat': '0',
    'dog': '1'
}


def generate_txt_file(txt_path, img_dir):
    """
    Generates a text file containing image paths and their labels.
    生成包含图像路径和标签的文本文件。

    Args:
        txt_path (str): The path to the output text file.
        img_dir (str): The root directory of the images.
    """
    # Use 'with' statement for safe file handling
    # 使用 'with' 语句，确保文件在处理完毕后自动关闭
    with open(txt_path, 'w') as f:
        # Traverse the subdirectories for 'cat' and 'dog'
        # 遍历 'cat' 和 'dog' 的子目录
        for class_name, label in CLASSES.items():
            class_dir = os.path.join(img_dir, class_name)

            # Check if the directory exists
            # 检查目录是否存在，增加代码的健壮性
            if not os.path.isdir(class_dir):
                print(f"Directory not found: {class_dir}")
                continue

            # Iterate through all files in the class directory
            # 遍历类别目录下的所有文件
            for img_file in os.listdir(class_dir):
                # Ensure the file is a jpg image
                # 确保文件是 .jpg 格式
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(class_dir, img_file)
                    line = f"{img_path} {label}\n"
                    f.write(line)

    print(f"Successfully generated {txt_path} with image paths and labels.")


if __name__ == '__main__':
    generate_txt_file(TRAIN_TXT_PATH, TRAIN_DIR)
    generate_txt_file(VALID_TXT_PATH, VALID_DIR)
