import os
import torch
import pandas as pd
from torch.optim import Adam
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertForSequenceClassification


class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()  # 删除缺失值

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


def prepare_dataloader():

    dataset = MyDataset()

    trainset, validset = random_split(
        dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42)
    )  # generator作用是设置随机数种子，保证每次划分数据集的结果一致

    tokenizer = BertTokenizer.from_pretrained("/gemini/code/model")

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(
            texts,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs["labels"] = torch.tensor(labels)
        return inputs

    trainloader = DataLoader(
        trainset,
        batch_size=32,
        collate_fn=collate_func,
        sampler=DistributedSampler(trainset),
    )  # DistributedSampler用于分布式训练，保证每个进程训练的数据不重复。为了保证数据大小一致，做额外的填充，评估指标可能会存在误差
    validloader = DataLoader(
        validset,
        batch_size=64,
        collate_fn=collate_func,
        sampler=DistributedSampler(validset),
    )

    return trainloader, validloader


def prepare_model_and_optimizer():

    model = BertForSequenceClassification.from_pretrained("/gemini/code/model")

    # LOCAL_RANK是进程的GPU编号，GLOBAL_RANK是进程的全局编号（多机），RANK是进程的本地编号
    if torch.cuda.is_available():
        model = model.to(int(os.environ["LOCAL_RANK"]))  # 作用是将模型放到指定GPU上

    model = DDP(model)  # DDP用于分布式训练

    optimizer = Adam(model.parameters(), lr=2e-5)

    return model, optimizer


def print_rank_0(info):
    if int(os.environ["RANK"]) == 0:
        print(info)


def evaluate(model, validloader):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {
                    k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()
                }
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    # dist.all_reduce用于将所有进程的acc_num相加，然后广播到所有进程
    dist.all_reduce(
        acc_num
    )  # 6种集合通信类型：Sactter、Gather、Reduce、AllReduce、Broadcast、AllGather
    return acc_num / len(validloader.dataset)


def train(model, optimizer, trainloader, validloader, epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        trainloader.sampler.set_epoch(ep)  # 作用是每个epoch都打乱数据集
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {
                    k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()
                }
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                dist.all_reduce(
                    loss, op=dist.ReduceOp.AVG
                )  # 作用是将所有进程的loss相加，然后求平均值
                print_rank_0(
                    f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}"
                )
            global_step += 1
        acc = evaluate(model, validloader)
        print_rank_0(f"ep: {ep}, acc: {acc}")  # 作用是只有进程0打印信息


def main():

    # 初始化进程组，backend是通信后端，这里使用nccl
    dist.init_process_group(backend="nccl")

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    train(model, optimizer, trainloader, validloader)


if __name__ == "__main__":
    main()


# torchrun --nproc_per_node=2 1.no_accelerate.py
