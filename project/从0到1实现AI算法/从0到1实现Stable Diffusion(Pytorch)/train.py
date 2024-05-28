from torch.utils.data import DataLoader
from data import MNIST
from add_noise import forward_add_noise
import torch
from torch import nn
import os
from unet import UNet

EPOCH=200
BATCH_SIZE=100      # 占用显存过大，调小这个
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 训练设备

def train():
    dataset=MNIST()
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)   # 数据加载器

    model=UNet(img_channel=1,channels=[64, 128, 256],time_emb_size=64,qsize=16,vsize=16,fsize=32,cls_emb_size=32).to(DEVICE)   # 噪音预测模型
    print(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3) # 优化器

    loss_fn=nn.L1Loss() # 损失函数(绝对值误差均值)

    model.train()

    iter_count=0
    for epoch in range(EPOCH):
        for imgs,labels in dataloader:
            x=imgs*2-1
            y=labels
            t=torch.randint(0,1000,(imgs.size(0),))
            x,noise=forward_add_noise(x,t)
            pred_noise=model(x.to(DEVICE),t.to(DEVICE),y.to(DEVICE))
            loss=loss_fn(pred_noise,noise.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_count%1000==0:
                print('epoch:{} iter:{},loss:{}'.format(epoch,iter_count,loss))
                torch.save(model.state_dict(),'stable_diffusion_mnist.pth.tmp')
                os.replace('stable_diffusion_mnist.pth.tmp','stable_diffusion_mnist.pth')
            iter_count+=1

if __name__ == '__main__':
    train()