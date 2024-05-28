import math
import torch
from torch import nn

class TimeEmbedding(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.half_emb_size=emb_size//2
        half_emb=torch.exp(torch.arange(self.half_emb_size)*(-1*math.log(10000)/(self.half_emb_size-1)))
        self.register_buffer('half_emb',half_emb)

    def forward(self,t):
        t=t.view(t.size(0),1)
        half_emb=self.half_emb.unsqueeze(0).expand(t.size(0),self.half_emb_size)
        half_emb_t=half_emb*t
        embs_t = torch.cat((half_emb_t.sin(), half_emb_t.cos()), dim=-1)
        return embs_t

class CrossAttention(nn.Module):
    def __init__(self,channel,qsize,vsize,fsize,cls_emb_size):
        super().__init__()
        self.w_q=nn.Linear(channel,qsize)
        self.w_k=nn.Linear(cls_emb_size,qsize)
        self.w_v=nn.Linear(cls_emb_size,vsize)
        self.softmax=nn.Softmax(dim=-1)
        self.z_linear=nn.Linear(vsize,channel)
        self.norm1=nn.LayerNorm(channel)
        # feed-forward结构
        self.feedforward=nn.Sequential(
            nn.Linear(channel,fsize),
            nn.ReLU(),
            nn.Linear(fsize,channel)
        )
        self.norm2=nn.LayerNorm(channel)

    def forward(self,x,cls_emb): # x:(batch_size,channel,width,height), cls_emb:(batch_size,cls_emb_size)
        x=x.permute(0,2,3,1) # x:(batch_size,width,height,channel)

        # 像素是Query
        Q=self.w_q(x)   # Q: (batch_size,width,height,qsize)
        Q=Q.view(Q.size(0),Q.size(1)*Q.size(2),Q.size(3))   # Q: (batch_size,width*height,qsize)

        # 引导分类是Key和Value
        K=self.w_k(cls_emb) # K: (batch_size,qsize)
        K=K.view(K.size(0),K.size(1),1) # K: (batch_size,qsize,1)
        V=self.w_v(cls_emb) # V: (batch_size,vsize)
        V=V.view(V.size(0),1,V.size(1))  # v: (batch_size,1,vsize)

        # 注意力打分矩阵Q*K
        attn=torch.matmul(Q,K)/math.sqrt(Q.size(2)) # attn: (batch_size,width*height,1)
        attn=self.softmax(attn) # attn: (batch_size,width*height,1)
        # print(attn) # 就一个Key&value，所以Query对其注意力打分总是1分满分

        # 注意力层的输出
        Z=torch.matmul(attn,V)    # Z: (batch_size,width*height,vsize)
        Z=self.z_linear(Z)  # Z: (batch_size,width*height,channel)
        Z=Z.view(x.size(0),x.size(1),x.size(2),x.size(3))   # Z: (batch_size,width,height,channel)

        # 残差&layerNorm
        Z=self.norm1(Z+x)# Z: (batch_size,width,height,channel)

        # FeedForward
        out=self.feedforward(Z)# Z: (batch_size,width,height,channel)
        # 残差&layerNorm
        out=self.norm2(out+Z)
        return out.permute(0,3,1,2)

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,time_emb_size,qsize,vsize,fsize,cls_emb_size):
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1), # 改通道数,不改大小
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        self.time_emb_linear=nn.Linear(time_emb_size,out_channel)    # Time时刻emb转成channel宽,加到每个像素点上
        self.relu=nn.ReLU()

        self.seq2=nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1), # 不改通道数,不改大小
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        # 像素做Query，计算对分类ID的注意力，实现分类信息融入图像，不改变图像形状和通道数
        self.crossattn=CrossAttention(channel=out_channel,qsize=qsize,vsize=vsize,fsize=fsize,cls_emb_size=cls_emb_size)

    def forward(self,x,t_emb,cls_emb): # t_emb: (batch_size,time_emb_size)
        x=self.seq1(x)  # 改通道数,不改大小
        t_emb=self.relu(self.time_emb_linear(t_emb)).view(x.size(0),x.size(1),1,1)   # t_emb: (batch_size,out_channel,1,1)
        output=self.seq2(x+t_emb)        # 不改通道数,不改大小
        return self.crossattn(output,cls_emb)   # 图像和引导向量做attention

class UNet(nn.Module):
    def __init__(self,img_channel=1,channels=[64, 128, 256, 512, 1024],time_emb_size=256,qsize=16,vsize=16,fsize=32,cls_emb_size=32):
        super().__init__()

        channels=[img_channel]+channels

        # time转embedding
        self.time_emb=nn.Sequential(
            TimeEmbedding(time_emb_size),
            nn.Linear(time_emb_size,time_emb_size),
            nn.ReLU(),
        )

        # 引导词cls转embedding
        self.cls_emb=nn.Embedding(10,cls_emb_size)

        # 每个encoder conv block增加一倍通道数
        self.enc_convs=nn.ModuleList()
        for i in range(len(channels)-1):
            self.enc_convs.append(ConvBlock(channels[i],channels[i+1],time_emb_size,qsize,vsize,fsize,cls_emb_size))

        # 每个encoder conv后马上缩小一倍图像尺寸,最后一个conv后不缩小
        self.maxpools=nn.ModuleList()
        for i in range(len(channels)-2):
            self.maxpools.append(nn.MaxPool2d(kernel_size=2,stride=2,padding=0))

        # 每个decoder conv前放大一倍图像尺寸，缩小一倍通道数
        self.deconvs=nn.ModuleList()
        for i in range(len(channels)-2):
            self.deconvs.append(nn.ConvTranspose2d(channels[-i-1],channels[-i-2],kernel_size=2,stride=2))

        # 每个decoder conv block减少一倍通道数
        self.dec_convs=nn.ModuleList()
        for i in range(len(channels)-2):
            self.dec_convs.append(ConvBlock(channels[-i-1],channels[-i-2],time_emb_size,qsize,vsize,fsize,cls_emb_size))   # 残差结构

        # 还原通道数,尺寸不变
        self.output=nn.Conv2d(channels[1],img_channel,kernel_size=1,stride=1,padding=0)

    def forward(self,x,t,cls):  # cls是引导词（图片分类ID）
        # time做embedding
        t_emb=self.time_emb(t)

        # cls做embedding
        cls_emb=self.cls_emb(cls)

        # encoder阶段
        residual=[]
        for i,conv in enumerate(self.enc_convs):
            x=conv(x,t_emb,cls_emb)
            if i!=len(self.enc_convs)-1:
                residual.append(x)
                x=self.maxpools[i](x)

        # decoder阶段
        for i,deconv in enumerate(self.deconvs):
            x=deconv(x)
            residual_x=residual.pop(-1)
            x=self.dec_convs[i](torch.cat((residual_x,x),dim=1),t_emb,cls_emb)    # 残差用于纵深channel维
        return self.output(x) # 还原通道数

if __name__=='__main__':
    unet=UNet(img_channel=1,channels=[64, 128, 256],time_emb_size=64,qsize=16,vsize=16,fsize=32,cls_emb_size=32)
    x=torch.rand(2,1,28,28)
    t=torch.randint(0,1000,(2,))
    y=torch.randint(0,10,(2,))
    outputs=unet(x,t,y)
    print(outputs.shape)