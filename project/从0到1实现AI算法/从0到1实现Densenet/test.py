import torch
from densenet import DenseNet
from data import MNIST
from torch.utils.data import DataLoader

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

def test():
    dataset=MNIST() # 数据集
    dataloader = DataLoader(dataset, batch_size=1000, num_workers=10, shuffle=False)

    model=DenseNet(init_channels=10, growth_rate=4, blocks=[6, 6, 6], num_classes=10).to(DEVICE) # 模型
    print(model)
    model.load_state_dict(torch.load('densenet_mnist.pth'))

    model.eval()    # 预测模式

    correct = 0
    total = 0

    with torch.no_grad(): # 在评估模式下不需要计算梯度
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # 模型预测
            logits = model(images)
            preds = logits.argmax(-1)

            # 计算准确率
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            accuracy = correct / total
            print(f'准确率: {accuracy * 100}%')

    print(f'最终准确率: {accuracy * 100}%')

if __name__ == '__main__':
    test()