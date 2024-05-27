from data import MNIST
import matplotlib.pyplot as plt
import torch
from vit import ViT
import torch.nn.functional as F
from torch.utils.data import DataLoader

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

def test():
    dataset=MNIST() # 数据集
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

    model=ViT(in_channels=1, patch_size=4, emb_size=64, img_size=28, num_classes=10, depth=3, n_heads=2, mlp_ratio=4.0, dropout=0.1).to(DEVICE) # 模型
    print(model)
    model.load_state_dict(torch.load('vit_mnist.pth'))

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