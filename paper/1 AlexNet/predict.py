import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# 检查是否有可用的CUDA设备，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义AlexNet模型架构 (必须与训练时保持一致)
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 类别标签
# 注意：这些标签必须与训练时使用的标签顺序一致
# # CIFAR-10
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# catdog
classes = ('cat', 'dog')


def predict_image(image_path, model_path):
    """
    加载模型并对单张图片进行预测。

    Args:
        image_path (str): 要预测的图片文件路径。
        model_path (str): 保存的模型文件路径。

    Returns:
        tuple: (预测类别, 预测概率)。
    """
    # 加载模型
    # # CIDAR-10
    # model = AlexNet(num_classes=10).to(device)
    # catdog
    model = AlexNet(num_classes=2).to(device)

    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return None, None

    # 加载和预处理图片
    try:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return None, None

    # 运行推理
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs.data, 1)
        predicted_class = classes[predicted_idx.item()]

        # 计算预测概率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted_idx.item()].item()

    return predicted_class, confidence


if __name__ == '__main__':
    # 你可以修改这里的图片路径，来测试你自己的图片
    image_to_predict = 'data/catVSdog/test_data/cat/cat.10000.jpg'

    # # CIFAR-10
    # model_file = 'alexnet_cifar10.pth'

    # catdog
    model_file = 'alexnet_catdog.pth'


    predicted_class, confidence = predict_image(image_to_predict, model_file)

    if predicted_class and confidence:
        print(f"The image is predicted as: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
