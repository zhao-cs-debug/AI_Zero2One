import time
import math
import torch
import pandas as pd

from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import BertTokenizer, BertForSequenceClassification

from peft import LoraConfig, get_peft_model


class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


def prepare_dataloader():

    dataset = MyDataset()

    trainset, validset = random_split(
        dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42)
    )

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
        trainset, batch_size=32, collate_fn=collate_func, shuffle=True
    )
    validloader = DataLoader(
        validset, batch_size=64, collate_fn=collate_func, shuffle=False
    )

    return trainloader, validloader


def prepare_model_and_optimizer():

    model = BertForSequenceClassification.from_pretrained("/gemini/code/model")

    lora_config = LoraConfig(
        target_modules=["query", "key", "value"]
    )  # 不指定任务类型会应用所有指定模块，指定任务类型会应用于指定任务类型的模块

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    optimizer = Adam(model.parameters(), lr=2e-5)

    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(validloader.dataset)


def train(
    model,
    optimizer,
    trainloader,
    validloader,
    accelerator: Accelerator,
    resume,
    epoch=3,
    log_step=10,
):
    global_step = 0
    start_time = time.time()

    resume_step = 0
    resume_epoch = 0

    if resume is not None:
        # 加载检查点
        accelerator.load_state(resume)
        # 计算每个epoch的步数
        steps_per_epoch = math.ceil(
            len(trainloader) / accelerator.gradient_accumulation_steps
        )
        # 从检查点中获取步数信息
        resume_step = global_step = int(resume.split("step_")[-1])
        # 计算跳过的轮数和步数
        resume_epoch = resume_step // steps_per_epoch
        resume_step -= resume_epoch * steps_per_epoch
        # 打印恢复的检查点日志
        accelerator.print(f"resume from checkpoint -> {resume}")

    for ep in range(resume_epoch, epoch):
        model.train()
        if resume and ep == resume_epoch and resume_step != 0:
            active_dataloader = accelerator.skip_first_batches(
                trainloader, resume_step * accelerator.gradient_accumulation_steps
            )
        else:
            active_dataloader = trainloader
        for batch in active_dataloader:
            with accelerator.accumulate(
                model
            ):  # 在训练循环中使用accumulate上下文管理器
                optimizer.zero_grad()
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                optimizer.step()
                # if global_step % log_step == 0:
                #     loss = accelerator.reduce(loss, "mean")
                #     accelerator.print(
                #         f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}"
                #     )
                # global_step += 1

                if accelerator.sync_gradients:  # 如果做了一次梯度同步
                    global_step += 1

                    if global_step % log_step == 0:
                        loss = accelerator.reduce(loss, "mean")
                        accelerator.print(
                            f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}"
                        )
                        accelerator.log({"loss": loss.item()}, global_step)  # 记录日志

                    if global_step % 50 == 0 and global_step != 0:
                        accelerator.print(f"save checkpoint -> step_{global_step}")
                        accelerator.save_state(
                            accelerator.project_dir + f"/step_{global_step}"
                        )  # 保存检查点
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=accelerator.project_dir
                            + f"/step_{global_step}/model",
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                            save_func=accelerator.save,
                        )  # 解包之后再保存模型，可以保存完整模型，也可以保存部分模型
        acc = evaluate(model, validloader, accelerator)
        # accelerator.print(f"ep: {ep}, acc: {acc}")
        accelerator.print(f"ep: {ep}, acc: {acc}, time: {time.time() - start_time}")
        accelerator.log({"acc": acc}, global_step)  # 记录日志

    accelerator.end_training()


def main():

    # accelerator = Accelerator()
    # 初始化Accelerator，设置梯度累积步数为2，使用tensorboard记录日志，模型保存在ckpts目录下
    accelerator = Accelerator(
        gradient_accumulation_steps=2, log_with="tensorboard", project_dir="ckpts"
    )

    # 初始化跟踪器，记录runs目录下的日志
    accelerator.init_trackers(project_name="runs")

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    model, optimizer, trainloader, validloader = accelerator.prepare(
        model, optimizer, trainloader, validloader
    )

    # train(model, optimizer, trainloader, validloader, accelerator)
    train(
        model,
        optimizer,
        trainloader,
        validloader,
        accelerator,
        resume="/gemini/code/ckpts/step_150",
    )


if __name__ == "__main__":
    main()

# 混合精度训练可以加速训练，但并不一定会降低显存占用。
# accelerator = Accelerator(mixed_precision="bf16")
# accelerator config && choice bf16
# accelerator launch --mixed_precision bf16 3.accelerate_advanced.py

# 梯度累计是允许模型在有限的硬件资源下模拟更大批量大小的训练效果
# accumulation_steps = 4  # 设定累计步数
# model.zero_grad()   # 清空梯度
# for step, (inputs, targets) in enumerate(data_loader):
#     outputs = model(inputs)
#     loss = loss_fn(outputs, targets)
#     loss = loss / accumulation_steps    # 缩放损失
#     loss.backward()                     # 计算梯度
#     if (step + 1) % accumulation_steps == 0:
#         optimizer.step()                # 更新参数
#         model.zero_grad()               # 清空梯度
# accelerator = Accelerator(gradient_accumulation_steps=xx) # 步骤一：创建Accelerator对象时指定梯度累积步数
# with accelerator.accumulate(model):                       # 步骤二：在训练循环中使用accumulate上下文管理器

# 实验记录工具：Tensorboard、Wandb
# accelerator = Accelerator(log_with="tensorboard", project_dir="xxx")  # 步骤一：使用Tensorboard记录日志，输出到xxx目录下
# accelerator.init_trackers(project_name="xxx")                         # 步骤二：初始化跟踪器，记录xxx目录下的日志
# accelerator.log(values: dict, step: int)                              # 步骤三：记录日志
# accelerator.end_training()                                            # 步骤四：结束训练，关闭所有tracker

# 模型保存内容
# 模型权重，pytorch_model.bin / model.safetensors
# 模型配置文件，关于模型结构描述的信息，一般是config.json
# 其他文件，generation_config.json、 adapter_model.safetensors
# 单机保存：model.save_pretrained(save_directory)
# 多机保存：model.save_pretrained(save_directory, is_main_process, state_dict, save_func)   # 主进程保存即可
# accelerator.save_model()                              # 方式一问题：不保存配置文件，只保存模型参数。对于Lora支持不好，会将完整模型保存，不会单独存一部分
# accelerator.unwrap_model(model).save_pretrained()     # 方式二：解包之后再保存模型，可以保存完整模型，也可以保存部分模型

# 断点续训允许从上次中断的地方恢复训练，而不是从头开始。
# 保存检查点
# 加载检查点（模型权重、优化器状态、学习率调度器、随机状态）
# 跳过已训练数据（epoch、batch）
# accelerator.save_state()                                  # 保存检查点
# accelerator.load_state()                                  # 加载检查点
# resume_epoch、resume_step                                 # 计算跳过的轮数和步数
# accelerator.skip_first_batches(trainloader, resume_step)  # 跳过已训练数据
