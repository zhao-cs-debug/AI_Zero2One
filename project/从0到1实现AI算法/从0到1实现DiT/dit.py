import math
import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x

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
        embs_t=torch.cat((half_emb_t.sin(),half_emb_t.cos()),dim=-1)
        return embs_t

class DiTBlock(nn.Module):
    def __init__(self,emb_size,nhead):
        super().__init__()

        self.emb_size=emb_size
        self.nhead=nhead

        # conditioning
        self.gamma1=nn.Linear(emb_size,emb_size)
        self.beta1=nn.Linear(emb_size,emb_size)
        self.alpha1=nn.Linear(emb_size,emb_size)
        self.gamma2=nn.Linear(emb_size,emb_size)
        self.beta2=nn.Linear(emb_size,emb_size)
        self.alpha2=nn.Linear(emb_size,emb_size)

        # layer norm
        self.ln1=nn.LayerNorm(emb_size)
        self.ln2=nn.LayerNorm(emb_size)

        # multi-head self-attention
        self.wq=nn.Linear(emb_size,nhead*emb_size) # (batch,seq_len,nhead*emb_size)
        self.wk=nn.Linear(emb_size,nhead*emb_size) # (batch,seq_len,nhead*emb_size)
        self.wv=nn.Linear(emb_size,nhead*emb_size) # (batch,seq_len,nhead*emb_size)
        self.lv=nn.Linear(nhead*emb_size,emb_size)

        # feed-forward
        self.ff=nn.Sequential(
            nn.Linear(emb_size,emb_size*4),
            nn.ReLU(),
            nn.Linear(emb_size*4,emb_size)
        )

    def forward(self,x,cond):   # x:(batch,seq_len,emb_size), cond:(batch,emb_size)
        # conditioning (batch,emb_size)
        gamma1_val=self.gamma1(cond)
        beta1_val=self.beta1(cond)
        alpha1_val=self.alpha1(cond)
        gamma2_val=self.gamma2(cond)
        beta2_val=self.beta2(cond)
        alpha2_val=self.alpha2(cond)

        # layer norm
        y=self.ln1(x) # (batch,seq_len,emb_size)

        # scale&shift
        y=y*(1+gamma1_val.unsqueeze(1))+beta1_val.unsqueeze(1)

        # attention
        q=self.wq(y)    # (batch,seq_len,nhead*emb_size)
        k=self.wk(y)    # (batch,seq_len,nhead*emb_size)
        v=self.wv(y)    # (batch,seq_len,nhead*emb_size)
        q=q.view(q.size(0),q.size(1),self.nhead,self.emb_size).permute(0,2,1,3) # (batch,nhead,seq_len,emb_size)
        k=k.view(k.size(0),k.size(1),self.nhead,self.emb_size).permute(0,2,3,1) # (batch,nhead,emb_size,seq_len)
        v=v.view(v.size(0),v.size(1),self.nhead,self.emb_size).permute(0,2,1,3) # (batch,nhead,seq_len,emb_size)
        attn=q@k/math.sqrt(q.size(2))   # (batch,nhead,seq_len,seq_len)
        attn=torch.softmax(attn,dim=-1)   # (batch,nhead,seq_len,seq_len)
        y=attn@v    # (batch,nhead,seq_len,emb_size)
        y=y.permute(0,2,1,3) # (batch,seq_len,nhead,emb_size)
        y=y.reshape(y.size(0),y.size(1),y.size(2)*y.size(3))    # (batch,seq_len,nhead*emb_size)
        y=self.lv(y)    # (batch,seq_len,emb_size)

        # scale
        y=y*alpha1_val.unsqueeze(1)
        # redisual
        y=x+y

        # layer norm
        z=self.ln2(y)
        # scale&shift
        z=z*(1+gamma2_val.unsqueeze(1))+beta2_val.unsqueeze(1)
        # feef-forward
        z=self.ff(z)
        # scale
        z=z*alpha2_val.unsqueeze(1)
        # residual
        return y+z


class DiT(nn.Module):
    def __init__(self,in_channels,patch_size,emb_size,img_size,label_num,dit_num,head):
        super().__init__()
        self.channel = in_channels
        self.patch_size = patch_size
        self.patch_count = img_size // patch_size
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, emb_size))

        # time emb
        self.time_emb=nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size,emb_size),
            nn.ReLU(),
            nn.Linear(emb_size,emb_size)
        )

        # label emb
        self.label_emb=nn.Embedding(num_embeddings=label_num,embedding_dim=emb_size)

        # DiT Blocks
        self.dits=nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size,head))

        # layer norm
        self.ln=nn.LayerNorm(emb_size)

        # linear back to patch
        self.linear=nn.Linear(emb_size,in_channels*patch_size**2)

    def forward(self,x,t,y): # x:(batch,channel,height,width)   t:(batch,)  y:(batch,)
        # label emb
        y_emb=self.label_emb(y) #   (batch,emb_size)
        # time emb
        t_emb=self.time_emb(t)  #   (batch,emb_size)

        # condition emb
        cond=y_emb+t_emb

        # patch emb
        x=self.patch_embedding(x)
        x+=self.positional_embedding

        # dit blocks
        for dit in self.dits:
            x=dit(x,cond)

        # layer norm
        x=self.ln(x)    #   (batch,patch_count**2,emb_size)

        # linear back to patch
        x=self.linear(x)    # (batch,patch_count**2,channel*patch_size*patch_size)

        # reshape
        x=x.view(x.size(0),self.patch_count,self.patch_count,self.channel,self.patch_size,self.patch_size)  # (batch,patch_count,patch_count,channel,patch_size,patch_size)
        x=x.permute(0,3,1,4,2,5)    # (batch,channel,patch_count(H),patch_size(H),patch_count(W),patch_size(W))
        x=x.reshape(x.size(0),self.channel,self.patch_count*self.patch_size,self.patch_count*self.patch_size)   # (batch,channel,img_size,img_size)
        return x

if __name__=='__main__':
    dit=DiT(in_channels=1,patch_size=4,emb_size=64,img_size=28,label_num=10,dit_num=3,head=4)
    x=torch.rand(5,1,28,28)
    t=torch.randint(0,1000,(5,))
    y=torch.randint(0,10,(5,))
    outputs=dit(x,t,y)
    print(outputs.shape)