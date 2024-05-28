import torch
from torch import nn
from unet import UNet
import matplotlib.pyplot as plt
from add_noise import *
from lora import *

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

def backward_denoise(model,x,y):
    steps=[x.clone(),]

    global alphas,alphas_cumprod,variance

    x=x.to(DEVICE)
    alphas=alphas.to(DEVICE)
    alphas_cumprod=alphas_cumprod.to(DEVICE)
    variance=variance.to(DEVICE)
    y=y.to(DEVICE)

    # BN层的存在，需要eval模式避免推理时跟随batch的数据分布，但是相反训练的时候需要更加充分让它见到各种batch数据分布
    model.eval()
    with torch.no_grad():
        for time in range(1000-1,-1,-1):
            t=torch.full((x.size(0),),time).to(DEVICE) #[999,999,....]

            # 预测x_t时刻的噪音
            noise=model(x,t,y)

            # 生成t-1时刻的图像
            shape=(x.size(0),1,1,1)
            mean=1/torch.sqrt(alphas[t].view(*shape))*  \
                (
                    x-
                    (1-alphas[t].view(*shape))/torch.sqrt(1-alphas_cumprod[t].view(*shape))*noise
                )
            if time!=0:
                x=mean+ \
                    torch.randn_like(x)* \
                    torch.sqrt(variance[t].view(*shape))
            else:
                x=mean
            x=torch.clamp(x, -1.0, 1.0).detach()
            steps.append(x)
    return steps

def test():
    # 加载模型
    model=UNet(img_channel=1,channels=[64, 128, 256],time_emb_size=64,qsize=16,vsize=16,fsize=32,cls_emb_size=32).to(DEVICE)
    model.load_state_dict(torch.load('stable_diffusion_mnist.pth'))

    USE_LORA=True

    if USE_LORA:
        # 向nn.Linear层注入Lora
        for name,layer in model.named_modules():
            name_cols=name.split('.')
            # 过滤出cross attention使用的linear权重
            filter_names=['w_q','w_k','w_v']
            if any(n in name_cols for n in filter_names) and isinstance(layer,nn.Linear):
                inject_lora(model,name,layer)

        # lora权重的加载
        try:
            restore_lora_state=torch.load('sd_lora.pth')
            model.load_state_dict(restore_lora_state,strict=False)
        except:
            Exception()

        model=model.to(DEVICE)

        # lora权重合并到主模型
        for name,layer in model.named_modules():
            name_cols=name.split('.')

            if isinstance(layer,LoraLayer):
                children=name_cols[:-1]
                cur_layer=model
                for child in children:
                    cur_layer=getattr(cur_layer,child)
                lora_weight=(layer.lora_a@layer.lora_b)*layer.alpha/layer.r
                before_weight=layer.raw_linear.weight.clone()
                layer.raw_linear.weight=nn.Parameter(layer.raw_linear.weight.add(lora_weight.T)).to(DEVICE)    # 把Lora参数加到base model的linear weight上
                setattr(cur_layer,name_cols[-1],layer.raw_linear)

    # 打印模型结构
    print(model)

    # 生成噪音图
    batch_size=10
    x=torch.randn(size=(batch_size,1,28,28))  # (5,1,28,28)
    y=torch.arange(start=0,end=10,dtype=torch.long)   # 引导词promot
    # 逐步去噪得到原图
    steps=backward_denoise(model,x,y)
    # 绘制数量
    num_imgs=20
    # 绘制还原过程
    plt.figure(figsize=(15,15))
    for b in range(batch_size):
        for i in range(0,num_imgs):
            idx=int(1000/num_imgs)*(i+1)
            # 像素值还原到[0,1]
            final_img=(steps[idx][b].to('cpu')+1)/2
            # tensor转回PIL图
            final_img=final_img.permute(1,2,0)
            plt.subplot(batch_size,num_imgs,b*num_imgs+i+1)
            plt.imshow(final_img)
    plt.show()

if __name__ == '__main__':
    test()