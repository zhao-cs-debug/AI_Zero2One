import torch
from data import MNIST
from vit import ViT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

EPOCH=10
BATCH_SIZE=64   # 从batch内选出10个不一样的数字
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

def train():
    dataset=MNIST() # 数据集
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)    # 数据加载器

    model=ViT(in_channels=1, patch_size=4, emb_size=64, img_size=28, num_classes=10, depth=3, n_heads=2, mlp_ratio=4.0, dropout=0.1).to(DEVICE) # 模型
    print(model)
    optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)   # 优化器

    model.train()

    iter_count=0
    for epoch in range(EPOCH):
        for imgs,labels in dataloader:
            logits=model(imgs.to(DEVICE))

            loss=F.cross_entropy(logits,labels.to(DEVICE))

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            if iter_count%1000==0:
                print('epoch:{} iter:{},loss:{}'.format(epoch,iter_count,loss))
                torch.save(model.state_dict(),'vit_mnist.pth.tmp')
                os.replace('vit_mnist.pth.tmp','vit_mnist.pth')
            iter_count+=1

if __name__ == '__main__':
    train()