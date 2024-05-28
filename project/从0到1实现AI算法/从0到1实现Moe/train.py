import os
import torch
from data import MNIST
from moe import MNIST_MoE
import torch.nn.functional as F
from torch.utils.data import DataLoader

EPOCH=10
BATCH_SIZE=64
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

def train():
    dataset=MNIST() # 数据集
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)    # 数据加载器

    model=MNIST_MoE(input_size=28*28,experts=8,top=2,emb_size=16).to(DEVICE) # 模型
    print(model)
    optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)   # 优化器

    model.train()

    iter_count=0
    for epoch in range(EPOCH):
        for imgs,labels in dataloader:
            logits,prob,imp_loss=model(imgs.to(DEVICE))

            loss=F.cross_entropy(logits,labels.to(DEVICE))
            loss=loss+imp_loss

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            if iter_count%1000==0:
                expert_stats=torch.argmax(prob,dim=-1).cpu().unique(return_counts=True)[1].numpy()
                print('epoch:{} iter:{},loss:{},experts:{}'.format(epoch,iter_count,loss,expert_stats))
                torch.save(model.state_dict(),'moe_mnist.pth.tmp')
                os.replace('moe_mnist.pth.tmp','moe_mnist.pth')
            iter_count+=1

if __name__ == '__main__':
    train()