import os
import torch
from torch import nn
from dit import DiT
from data import MNIST
from torch.utils.data import DataLoader
from add_noise import *

EPOCH=500
BATCH_SIZE=300
DEVICE='cuda' if torch.cuda.is_available() else 'cpu' # 设备

def train():
    dataset=MNIST() # 数据集
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)    # 数据加载器

    model=DiT(in_channels=1,patch_size=4,emb_size=64,img_size=28,label_num=10,dit_num=3,head=4).to(DEVICE) # 模型
    print(model)
    optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)   # 优化器

    loss_fn=nn.L1Loss() # 损失函数(绝对值误差均值)

    model.train()

    iter_count=0
    for epoch in range(EPOCH):
        for imgs,labels in dataloader:
            x=imgs*2-1 # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应
            y=labels
            t=torch.randint(0,1000,(imgs.size(0),))  # 为每张图片生成随机t时刻
            x,noise=forward_add_noise(x,t) # x:加噪图 noise:噪音
            pred_noise=model(x.to(DEVICE),t.to(DEVICE),y.to(DEVICE))
            loss=loss_fn(pred_noise,noise.to(DEVICE))
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            if iter_count%1000==0:
                print('epoch:{} iter:{},loss:{}'.format(epoch,iter_count,loss))
                torch.save(model.state_dict(),'dit_mnist.pth.tmp')
                os.replace('dit_mnist.pth.tmp','dit_mnist.pth')
            iter_count+=1

if __name__ == '__main__':
    train()