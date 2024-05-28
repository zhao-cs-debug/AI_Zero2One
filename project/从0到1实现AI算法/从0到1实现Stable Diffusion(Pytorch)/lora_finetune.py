import os
import torch
from torch import nn
from unet import UNet
from data import MNIST
from torch.utils.data import DataLoader
from add_noise import *
from lora import *

EPOCH=20
BATCH_SIZE=100
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def lora_train():
    dataset=MNIST()
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)   # 数据加载器

    model=UNet(img_channel=1,channels=[64, 128, 256],time_emb_size=64,qsize=16,vsize=16,fsize=32,cls_emb_size=32).to(DEVICE)   # 噪音预测模型
    model.load_state_dict(torch.load('stable_diffusion_mnist.pth'))
    print(model)

    # 向nn.Linear层注入Lora
    for name,layer in model.named_modules():
        name_cols=name.split('.')
        # 过滤出cross attention使用的linear权重
        filter_names=['w_q','w_k','w_v']
        if any(n in name_cols for n in filter_names) and isinstance(layer,nn.Linear):
            inject_lora(model,name,layer)

    # 冻结非Lora参数
    for name,param in model.named_parameters():
        if name.split('.')[-1] not in ['lora_a','lora_b']:  # 非Lora部分不计算梯度
            param.requires_grad=False
        else:
            param.requires_grad=True

    print(model)
    model=model.to(DEVICE)
    optimizer=torch.optim.Adam(filter(lambda x: x.requires_grad==True,model.parameters()),lr=1e-3) # 优化器只更新Lorac参数

    loss_fn=nn.L1Loss() # 损失函数(绝对值误差均值)

    model.train()

    iter_count=0
    for epoch in range(EPOCH):
        for imgs,labes in dataloader:
            x=imgs*2-1
            y=labes
            t=torch.randint(0,1000,(imgs.size(0),))
            x,noise=forward_add_noise(x,t)
            pred_noise=model(x.to(DEVICE),t.to(DEVICE),y.to(DEVICE))
            loss=loss_fn(pred_noise,noise.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_count%1000==0:
                print('epoch:{} iter:{},loss:{}'.format(epoch,iter_count,loss))
                # 保存训练好的Lora权重
                lora_state={}
                for name,param in model.named_parameters():
                    name_cols=name.split('.')
                    filter_names=['lora_a','lora_b']
                    if any(n==name_cols[-1] for n in filter_names):
                        lora_state[name]=param
                torch.save(lora_state,'sd_lora.pth.tmp')
                os.replace('sd_lora.pth.tmp','sd_lora.pth')
            iter_count+=1

if __name__ == '__main__':
    lora_train()
