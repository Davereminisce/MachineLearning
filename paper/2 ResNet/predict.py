import torch
from torchvision import transforms
from PIL import Image

# 导入我的模型架构
from ResNet import ResNet34  # 我的模型 ResNet34 定义在 ResNet.py 文件中


# CIFAR-10 类别标签
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ======== 配置模型路径和测试图片路径 ========
model_path = "model/resnet34_final_20.pth"  # 训练好的模型路径
image_path = "cat.10002.jpg"  # 预测图片路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================

def load_image(image_path):
    """
    加载图片并进行预处理
    """
    transform = transforms.Compose([
        transforms.Resize(224),  # 调整为模型输入尺寸
        transforms.ToTensor(),  # 转为 Tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    image = Image.open(image_path).convert('RGB')  # 确保是 RGB 图
    image = transform(image).unsqueeze(0)  # 增加 batch 维度
    return image


def predict(image_path, model_path, device):
    """
    对单张图片进行预测
    """
    print(f"Using device: {device}")

    # 加载模型
    model = ResNet34(num_classes=10)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 加载图片
    image = load_image(image_path).to(device)

    # 前向传播
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        class_idx = predicted.item()
        class_name = classes[class_idx]

    return class_name


if __name__ == "__main__":
    class_name = predict(image_path, model_path, device)
    print(f"Predicted class for '{image_path}': {class_name}")

