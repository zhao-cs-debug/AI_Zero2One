import torch
from data import MNIST
from moe import MNIST_MoE
from torch.utils.data import DataLoader

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

def test():
    dataset=MNIST() # 数据集
    dataloader=DataLoader(dataset,batch_size=1000,num_workers=10,persistent_workers=True)    # 数据加载器

    model=MNIST_MoE(input_size=28*28,experts=8,top=2,emb_size=16).to(DEVICE) # 模型
    print(model)
    model.load_state_dict(torch.load('moe_mnist.pth'))

    model.eval()    # 预测模式

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # 模型预测
            logits,_,_ = model(images)
            preds = logits.argmax(-1)

            # 计算准确率
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            accuracy = correct / total
            print(f'准确率: {accuracy * 100}%')

    print(f'最终准确率: {accuracy * 100}%')

if __name__ == '__main__':
    test()