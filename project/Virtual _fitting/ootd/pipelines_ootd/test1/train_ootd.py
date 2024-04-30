#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse  # 用于解析命令行参数
import logging  # 用于生成日志
import math  # 提供基本数学功能
import os  # 提供了丰富的方法来处理文件和目录
import random  # 生成伪随机数
import shutil  # 文件和文件夹的高级操作，例如复制和删除
from pathlib import Path  # 提供面向对象的文件系统路径
from collections import defaultdict  # 提供带有默认值的字典

import accelerate  # Huggingface的加速库，用于简化模型加速和分布式运行
import numpy as np  # 科学计算的基础包
import torch  # PyTorch，一个深度学习框架
import torch.nn.functional as F  # PyTorch的函数接口
import torch.utils.checkpoint  # 用于在PyTorch中实现梯度检查点
import transformers  # Huggingface的transformers库，用于自然语言处理
from accelerate import Accelerator  # 加速库的主要组件，用于简化模型加速
from accelerate.logging import get_logger  # 从accelerate库获取日志功能
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)  # 加速库的工具函数
from datasets import load_dataset  # Huggingface的datasets库，用于加载数据集
from huggingface_hub import (
    create_repo,
    upload_folder,
)  # Huggingface的hub库，用于创建和上传模型
from packaging import version  # 用于处理版本号
from PIL import Image  # Python的图像处理库
from torchvision import transforms  # PyTorch的图像处理库
from tqdm.auto import tqdm  # 进度条库
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
)  # 自动加载tokenizer和预训练配置

import diffusers  # Huggingface的diffusers库，用于训练稳定扩散模型
from diffusers import (  # diffusers库的主要组件
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler  # 获取扩散过程的调度器
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)  # diffusers库的工具函数
from diffusers.utils.hub_utils import (
    load_or_create_model_card,
    populate_model_card,
)  # 模型卡片处理
from diffusers.utils.import_utils import is_xformers_available  # 检查特定库是否可用
from diffusers.utils.torch_utils import is_compiled_module  # 检查模块是否已编译
from diffusers.utils import (  # diffusers库的工具函数
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    deprecate,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from transformers import (
    AutoProcessor,
    CLIPVisionModelWithProjection,
)  # 自动加载处理器和CLIP视觉模型
from transformers import CLIPTextModel, CLIPTokenizer  # CLIP文本模型和tokenizer
from diffusers.image_processor import (
    PipelineImageInput,
    VaeImageProcessor,
)  # 图像处理器

from data_scripts.cp_dataset import CPDatasetV2 as CPDataset  # CP数据集
import wandb  # 用于记录实验结果

from ootd.train_ootd_hd import OOTDiffusionModel  # 自定义的扩散模型训练脚本
from ootd.pipelines_ootd.pipeline_ootd import (
    OotdPipeline as OotdPipelineInference,
)  # 自定义的推理管道

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.27.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols  # 确保图像的数量正好填满网格

    w, h = imgs[0].size  # 获取单个图像的宽度和高度
    grid = Image.new(
        "RGB", size=(cols * w, rows * h)
    )  # 创建一个新的大图像，用于放置所有小图像

    for i, img in enumerate(imgs):  # 遍历每个小图像
        grid.paste(img, box=(i % cols * w, i // cols * h))  # 将图像粘贴到正确的位置
    return grid  # 返回组合后的大图像


def log_validation(
    model,
    args,
    accelerator,
    weight_dtype,
    test_dataloder=None,
    validation_dataloader=None,
):
    logger.info("Running validation... ")

    # 解包模型中的部分网格
    unet_garm = accelerator.unwrap_model(model.unet_garm)
    unet_vton = accelerator.unwrap_model(model.unet_vton)

    # 从预训练模型创建一个推理pipeline
    pipeline = OotdPipelineInference.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        unet_garm=unet_garm,
        unet_vton=unet_vton,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    # 如果启用了xformers的内存高效注意力机制，则激活该功能
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    # 设置随机数生成器
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    def sample_imgs(data_loader, log_key: str):
        image_logs = []
        with torch.no_grad():  # 不计算梯度
            for _, batch in enumerate(data_loader):
                with torch.autocast("cuda"):
                    # 从batch中提取数据
                    prompt = batch["prompt"][0]
                    image_garm = batch["ref_imgs"][0, :]
                    image_vton = batch["inpaint_image"][0, :]
                    image_ori = batch["GT"][0, :]
                    inpaint_mask = batch["inpaint_mask"][0, :]
                    mask = batch["mask"][0, :].unsqueeze(0)

                    # what is this doing?
                    # 处理图像和文本的嵌入
                    prompt_image = model.auto_processor(
                        images=image_garm, return_tensors="pt"
                    ).to(accelerator.device)
                    prompt_image = model.image_encoder(
                        prompt_image.data["pixel_values"]
                    ).image_embeds
                    prompt_image = prompt_image.unsqueeze(1)
                    prompt_embeds = model.text_encoder(
                        model.tokenize_captions([prompt], 2).to(accelerator.device)
                    )[0]
                    prompt_embeds[:, 1:] = prompt_image[:]

                    # 使用pipeline生成图像
                    samples = pipeline(
                        prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton,
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=args.inference_steps,
                        generator=generator,
                    ).images[0]

                    # 记录生成的图像和其他信息
                    image_logs.append(
                        {
                            "garment": image_garm,
                            "model": image_vton,
                            "orig_img": image_ori,
                            "samples": samples,
                            "prompt": prompt,
                            "inpaint mask": inpaint_mask,
                            "mask": mask,
                        }
                    )

        # 使用wandb等工具记录和展示图像
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                formatted_images = []
                for log in image_logs:
                    # 将不同类型的图像和标注添加到日志中
                    formatted_images.append(
                        wandb.Image(log["garment"], caption="garment images")
                    )
                    formatted_images.append(
                        wandb.Image(log["model"], caption="masked model images")
                    )
                    formatted_images.append(
                        wandb.Image(log["orig_img"], caption="original images")
                    )
                    formatted_images.append(
                        wandb.Image(log["inpaint mask"], caption="inpaint mask")
                    )
                    formatted_images.append(wandb.Image(log["mask"], caption="mask"))
                    formatted_images.append(
                        wandb.Image(log["samples"], caption=log["prompt"])
                    )
                tracker.log(
                    {log_key: formatted_images}
                )  # 将格式化后的图像日志上传到wandb
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

    # 如果有验证数据集，处理并记录验证图像
    if validation_dataloader is not None:
        sample_imgs(validation_dataloader, "validation_images")
    # 如果有测试数据集，处理并记录测试图像
    if test_dataloder is not None:
        sample_imgs(test_dataloder, "test_images")


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    # 使用预训练模型名称或路径加载配置，指定修订版本和子文件夹
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    # 从配置中获取模型的架构类型
    model_class = text_encoder_config.architectures[0]

    # 根据模型的架构类型，导入相应的模型类
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    # TODO: what hell is this?
    # 如果是特定的模型类型，则从对应的库中导入
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    # 如果模型架构不支持，则抛出错误
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    """Only used for pushing the model HF hub."""
    img_str = ""
    # 如果提供了图像日志，处理并生成图像和文本描述
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            # 保存主控制图片
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            # 保存图片网格
            image_grid(images, 1, len(images)).save(
                os.path.join(repo_folder, f"images_{i}.png")
            )
            # 更新模型卡片的图像部分
            img_str += f"![images_{i})](./images_{i}.png)\n"

    # 构建模型描述字符串
    model_description = f"""controlnet-{repo_id}
    These are controlnet weights trained on {base_model} with new type of conditioning.{img_str}"""
    # 加载或创建模型卡片
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    # 设置模型卡片的标签
    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
    ]
    # 填充模型卡片的标签
    model_card = populate_model_card(model_card, tags=tags)

    # 保存模型卡片到指定目录
    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(
    input_args=None,
):  # 定义一个函数 parse_args，它接受一个可选参数 input_args
    parser = argparse.ArgumentParser(
        description="Simple example of a ControlNet training script."
    )  # 创建一个 ArgumentParser 对象，提供脚本的描述信息

    parser.add_argument(
        "--model_type",
        type=str,
        default="hd",
        help="We will have two types of models, half body and full body.",
    )  # 添加一个命令行参数 --model_type，指定模型类型，默认值为 "hd"

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )  # 添加一个必需的命令行参数 --pretrained_model_name_or_path，指定预训练模型的路径或名称

    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )  # 添加一个命令行参数 --controlnet_model_name_or_path，指定 ControlNet 模型的路径或名称，如果未指定则从 unet 初始化权重
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )  # 添加一个命令行参数 --revision，用于指定预训练模型的修订版本
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )  # 添加一个命令行参数 --variant，用于指定预训练模型的变体，例如 fp16
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )  # 添加一个命令行参数 --tokenizer_name，用于指定分词器的名称或路径，如果它与模型名称不同
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )  # 添加一个命令行参数 --output_dir，指定输出目录，模型预测和检查点将被写入此目录
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )  # 添加一个命令行参数 --cache_dir，指定缓存目录，下载的模型和数据集将被存储在此目录
    parser.add_argument(
        "--seed", type=int, default=-1, help="A seed for reproducible training."
    )  # 添加一个命令行参数 --seed，用于设置随机种子，以实现可重复的训练
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )  # 添加一个命令行参数 --resolution，指定输入图像的分辨率，训练/验证数据集中的所有图像都将被调整到这个分辨率
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )  # 添加一个命令行参数 --train_batch_size，指定训练数据加载器的批量大小（每个设备）
    parser.add_argument(
        "--num_train_epochs", type=int, default=1
    )  # 添加一个命令行参数 --num_train_epochs，指定训练的总周期数，默认为1
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )  # 添加一个命令行参数 --max_train_steps，指定要执行的训练步数总数。如果提供了此参数，它将覆盖 --num_train_epochs 的设置。
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )  # 添加一个命令行参数 --checkpointing_steps，指定每 X 次更新保存一次训练状态的检查点。检查点可用于通过 `--resume_from_checkpoint` 恢复训练。
    # 如果检查点比最终训练的模型更好，还可以用于推理。使用检查点进行推理需要单独加载原始管道和各个检查点模型组件。具体步骤可参考链接中的指导。
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )  # 添加一个命令行参数 `--checkpoints_total_limit`，指定最大的检查点（模型保存点）数量。  帮助信息：存储的最大检查点数。
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )  # 添加一个命令行参数 `--resume_from_checkpoint`，用于指定是否从之前的检查点恢复训练。 帮助信息：是否应从先前的检查点恢复训练。使用由 `--checkpointing_steps` 保存的路径，或者使用 `"latest"` 自动选择最新的检查点。
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )  # 添加一个命令行参数 `--gradient_accumulation_steps`，指定在执行反向更新（backward/update）前累积的更新步数。   帮助信息：执行反向/更新传递前要累积的更新步数。
    parser.add_argument(
        "--gradient_checkpointing_garm",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )  # 添加一个命令行参数 `--gradient_checkpointing_garm`，用于开启或关闭梯度检查点（节省内存但增加计算时间）。  帮助信息：是否使用梯度检查点来节省内存，尽管会使反向传递变慢。
    parser.add_argument(
        "--gradient_checkpointing_vton",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )  # 添加一个类似的命令行参数 `--gradient_checkpointing_vton`。
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )  # 添加一个命令行参数 `--learning_rate`，设置初始学习率。    帮助信息：使用的初始学习率（在潜在的预热期之后）。
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )  # 添加一个命令行参数 `--scale_lr`，用于根据 GPU 数量、梯度累积步骤和批量大小来缩放学习率。  帮助信息：通过 GPU 数量、梯度累积步骤和批量大小来缩放学习率。
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )  # 添加一个命令行参数 `--lr_scheduler`，选择使用的学习率调度器类型。  帮助信息：要使用的调度程序类型。 选择 ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"] 之间。
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )  # 添加一个命令行参数 `--lr_warmup_steps`，指定学习率预热期的步数。  帮助信息：学习率调度程序中预热的步数。
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )  # 添加一个命令行参数 `--lr_num_cycles`，用于指定 `cosine_with_restarts` 调度器中的重启次数。  帮助信息：`cosine_with_restarts` 调度程序中的硬重置次数。
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )  # 添加一个命令行参数 `--lr_power`，用于设置多项式调度器中的幂因子。  帮助信息：多项式调度程序的幂因子。
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )  # 添加一个命令行参数 `--use_8bit_adam`，是否使用 8-bit 版本的 Adam 优化器。  帮助信息：是否使用 bitsandbytes 提供的 8 位 Adam 优化器。
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )  # 添加一个命令行参数 `--dataloader_num_workers`，设置数据加载时使用的子进程数。  帮助信息：用于数据加载的子进程数。0 表示数据将在主进程中加载。
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )  # 添加一个命令行参数 `--adam_beta1`，设置 Adam 优化器中的 beta1 参数。  帮助信息：Adam 优化器的 beta1 参数。
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )  # 添加一个命令行参数 `--adam_beta2`，设置 Adam 优化器中的 beta2 参数。  帮助信息：Adam 优化器的 beta2 参数。
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )  # 添加一个命令行参数 `--adam_weight_decay`，设置 Adam 优化器中使用的权重衰减。  帮助信息：要使用的权重衰减。
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )  # 添加一个命令行参数 `--adam_epsilon`，设置 Adam 优化器中的 epsilon 参数。  帮助信息：Adam 优化器的 epsilon 值。
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )  # 添加一个命令行参数 --max_grad_norm，用于设置梯度的最大范数，默认值为1.0。 帮助信息：最大梯度范数。
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )  # 添加一个命令行参数 --push_to_hub，用于决定是否将模型推送到Hub。 帮助信息：是否将模型推送到Hub。
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )  # 添加一个命令行参数 --hub_token，用于设置推送到模型Hub的令牌。 帮助信息：用于推送到模型Hub的令牌。
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )  # 添加一个命令行参数 --hub_model_id，用于设置与本地output_dir保持同步的仓库名称。 帮助信息：要与本地 `output_dir` 同步的存储库的名称。
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )  # 添加一个命令行参数 --logging_dir，用于设置TensorBoard的日志目录，默认为logs。 帮助信息：[TensorBoard](https://www.tensorflow.org/tensorboard) 日志目录。默认为 *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***。
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )  # 添加一个命令行参数 --allow_tf32，用于决定是否在Ampere GPU上允许使用TF32，可以加速训练。 帮助信息：是否在Ampere GPU上允许使用TF32。可以用于加速训练。
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )  # 添加一个命令行参数 --report_to，用于设置结果和日志报告的平台，默认为tensorboard。 帮助信息：报告结果和日志的集成。支持的平台有 `"tensorboard"`（默认）、`"wandb"` 和 `"comet_ml"`。使用 `"all"` 报告所有集成。
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )  # 添加一个命令行参数 --mixed_precision，用于设置是否使用混合精度训练，可选值为"no"、"fp16"和"bf16"。    帮助信息：是否使用混合精度。在 fp16 和 bf16（bfloat16）之间选择。Bf16 需要 PyTorch >= 1.10 和 Nvidia Ampere GPU。默认为当前系统的加速配置或使用 `accelerate.launch` 命令传递的标志的值。使用此参数覆盖加速配置。
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )  # 添加一个命令行参数 --enable_xformers_memory_efficient_attention，用于决定是否使用xformers。 帮助信息：是否使用 xformers。
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )  # 添加一个命令行参数 --set_grads_to_none，用于在设置梯度为None而不是0来节省内存。 帮助信息：通过将梯度设置为 None 而不是零来节省更多内存。请注意，这会改变某些行为，如果它引起任何问题，请禁用此参数。更多信息：https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )  # 添加一个命令行参数 --dataset_name，用于设置训练使用的数据集名称或路径。 帮助信息：要训练的数据集的名称（来自 HuggingFace hub），也可以是指向文件系统中数据集的本地副本的路径，或者指向包含 🤗 Datasets 可以理解的文件的文件夹。
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )  # 添加一个命令行参数 --dataset_config_name，用于设置数据集的配置，如果数据集只有一个配置则可以不设置。 帮助信息：数据集的配置，如果只有一个配置则不设置。
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )  # 添加一个命令行参数 --train_data_dir，用于设置训练数据所在的文件夹路径。 帮助信息：包含训练数据的文件夹。文件夹内容必须遵循 https://huggingface.co/docs/datasets/image_dataset#imagefolder 中描述的结构。特别是，必须存在一个 `metadata.jsonl` 文件来为图像提供标题。如果指定了 `dataset_name`，则忽略此参数。
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image.",
    )  # 添加一个命令行参数 --image_column，用于指定包含目标图像的数据集列名，默认为'image'。 帮助信息：包含目标图像的数据集列。

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )  # 添加一个整型参数 `--max_train_samples`，用于调试或加速训练时，限制训练样本的数量。  帮助信息：为了调试或加快训练速度，如果设置了此值，则将训练示例的数量截断为此值。
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )  # 添加一个浮点型参数 `--proportion_empty_prompts`，用于设置被替换为空字符串的图片提示的比例，默认为0（不替换）。  帮助信息：要替换为空字符串的图像提示的比例。默认为0（不替换）。

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )  # 添加一个整型参数 `--num_validation_images`，用于设置每对验证图片和提示生成的图片数量，默认为4。  帮助信息：为每个 `--validation_image`、`--validation_prompt` 对生成的图像数量。
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )  # 添加一个整型参数 `--validation_steps`，用于设置每多少步骤进行一次验证，默认为100。  帮助信息：每 X 步运行一次验证。验证包括多次运行提示 `args.validation_prompt`：`args.num_validation_images` 并记录图像。
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )  # 添加一个字符串参数 `--tracker_project_name`，用于设置跟踪器项目的名称，默认为"train_controlnet"。  帮助信息：传递给 Accelerator.init_trackers 的 `project_name` 参数，更多信息请参见 https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator

    parser.add_argument(
        "--dataroot",
        type=str,
        default=None,
    )  # 添加一个字符串参数 `--dataroot`，用于指定数据的根目录，默认为None。

    parser.add_argument(
        "--train_data_list",
        type=str,
        default=None,
    )  # 添加一个字符串参数 `--train_data_list`，用于指定训练数据列表的路径，默认为None。

    parser.add_argument(
        "--validation_data_list",
        type=str,
        default=None,
    )  # 添加一个字符串参数 `--validation_data_list`，用于指定验证数据列表的路径

    parser.add_argument(
        "--test_data_list",
        type=str,
        default=None,
    )  # 添加一个字符串参数 `--test_data_list`，用于指定测试数据列表的路径

    # TODO: How to set up for multiple GPUs?
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )  # 添加一个整型参数 `--gpu_id`，用于指定使用的GPU的ID

    parser.add_argument(
        "--inference_steps",
        type=int,
        default=20,
    )  # 添加一个整型参数 `--inference_steps`，用于设置推断步骤的次数

    parser.add_argument(
        "--log_grads",
        action="store_true",
        help="Whether log the gradients of trained parts.",
    )  # 添加一个布尔型参数 `--log_grads`，用于设置是否记录训练部分的梯度

    parser.add_argument(
        "--vit_path",
        type=str,
        default="openai/clip-vit-large-patch14",
    )  # 添加一个字符串参数 `--vit_path`，用于指定ViT模型的路径

    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
    )  # 添加一个字符串参数 `--vae_path`，用于指定VAE模型的路径

    parser.add_argument(
        "--unet_path",
        type=str,
        default=None,
    )  # 添加一个字符串参数 `--unet_path`，用于指定U-Net模型的路径

    parser.add_argument(
        "--notes",
        type=str,
        default="",
    )  # 添加一个字符串参数 `--notes`，用于记录额外的备注信息

    parser.add_argument(
        "--tracker_entity",
        type=str,
        default="catchonlabs",
    )  # 添加一个字符串参数 `--tracker_entity`，用于设置跟踪器的实体

    parser.add_argument(
        "--clip_grad_norm",
        action="store_true",
        help="if clip the gradients' norm by max_grad_norm",
    )  # 添加一个布尔型参数 `--clip_grad_norm`，用于设置是否通过最大梯度范数来剪辑梯度。

    parser.add_argument(
        "--vton_unet_path",
        type=str,
        default=None,
    )  # 添加一个字符串参数 --vton_unet_path，用于指定虚拟试衣 U-Net 模型的路径。

    parser.add_argument(
        "--garm_unet_path",
        type=str,
        default=None,
    )  # 添加一个字符串参数 --garm_unet_path，用于指定服装 U-Net 模型的路径。

    # 判断是否有输入的命令行参数，如果有则解析输入的参数，否则解析命令行输入的参数。
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # 如果未指定数据集名称 (--dataset_name) 且未指定训练数据目录 (--train_data_dir)，抛出异常。
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    # 如果同时指定了数据集名称和训练数据目录，抛出异常。
    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    # 如果 --proportion_empty_prompts 的值不在 0 到 1 的范围内，抛出异常。
    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    # 如果 --resolution 不是 8 的倍数，抛出异常。这是为了确保在 VAE 和控制网编码器之间的编码图像大小一致。
    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args  # 返回解析后的参数对象。


def main(args):
    # 设置标记，这里的标记是训练来自OOTDiffusion模型。
    args.notes = "Train from OOTDiffusion model."
    # 如果同时使用了wandb和hub_token，则抛出错误，因为这样做可能暴露令牌，造成安全风险。
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # 设置日志目录。
    logging_dir = Path(args.output_dir, args.logging_dir)

    # 配置加速器项目。
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    # 分布式训练的相关参数设置。
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    # 设置日志参数，用于调试。
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # 记录日志信息。
    logger.info(accelerator.state, main_process_only=False)
    # 根据是否是主进程，设置日志级别。
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # 如果提供了seed，则设置随机种子，保证实验可复现性。
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    # 如果是主进程，并且指定了输出目录，则创建目录。
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # 如果需要推送到hub，创建仓库。
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizer
    # TODO: create tokenizer in OOTD model
    # 加载tokenizer。
    if args.tokenizer_name:
        tokenizer_path = args.tokenizer_name
    elif args.pretrained_model_name_or_path:
        tokenizer_path = args.pretrained_model_name_or_path

    # 设置模型的路径。
    if args.vton_unet_path is None:
        vton_unet_path = args.pretrained_model_name_or_path
    else:
        vton_unet_path = args.vton_unet_path

    if args.garm_unet_path is None:
        garm_unet_path = args.pretrained_model_name_or_path
    else:
        garm_unet_path = args.garm_unet_path

    # Load scheduler and models
    # 根据模型类型加载模型。
    if args.model_type == "hd":
        # TODO: it is better to move all these paths to args or a config file.
        model = OOTDiffusionModel(
            accelerator.device,
            model_path=args.pretrained_model_name_or_path,
            vton_unet_path=f"{vton_unet_path}",
            garm_unet_path=f"{garm_unet_path}",
            vit_path=args.vit_path,
        )
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")

    # `accelerate` 0.16.0 will have better support for customized saving
    # 版本检查，如果accelerate版本高于或等于0.16.0，使用自定义的模型保存和加载钩子。
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(
                    input_dir, subfolder="controlnet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 如果启用了xformers的内存高效注意力机制
    if args.enable_xformers_memory_efficient_attention:
        # 检查xformers库是否已安装并可以导入
        if is_xformers_available():
            import xformers  # 导入xformers库

            # 获取xformers的版本信息，并与特定版本号进行比较
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                # 如果版本是0.0.16，记录一条警告日志
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            # 启用两个模型组件的xformers内存高效注意力机制
            model.unet_garm.enable_xformers_memory_efficient_attention()
            model.unet_vton.enable_xformers_memory_efficient_attention()
        else:
            # 如果xformers库不可用，则抛出错误
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Check that all trainable models are in full precision
    # 检查所有可训练模型是否都使用了完整精度
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # if unwrap_model(controlnet).dtype != torch.float32:
    #     raise ValueError(
    #         f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
    #     )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # 如果允许使用TF32，在Ampere GPU上启用以加速训练
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 如果启用学习率缩放
    if args.scale_lr:
        # 调整学习率
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    # 如果使用8比特的Adam优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb  # 尝试导入bitsandbytes库
        except ImportError:
            # 如果库导入失败，抛出错误
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit  # 使用8比特的AdamW优化器
    else:
        optimizer_class = torch.optim.AdamW  # 否则使用标准的AdamW优化器

    # Optimizer creation
    # TODO: use args to control trainning para
    # params_to_optimize = (list(model.unet_garm.parameters()) + list(model.unet_vton.parameters()) + list(model.vae.parameters()))

    # 根据是否启用了梯度检查点来设置优化的参数
    if args.gradient_checkpointing_vton and args.gradient_checkpointing_garm:
        params_to_optimize = list(model.unet_garm.parameters()) + list(
            model.unet_vton.parameters()
        )
    elif args.gradient_checkpointing_vton:
        params_to_optimize = list(model.unet_vton.parameters())
    elif args.gradient_checkpointing_garm:
        params_to_optimize = list(model.unet_garm.parameters())

    # 创建优化器实例
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 检查数据根目录是否设置
    if args.dataroot is None:
        assert "Please provide correct data root"
    # 创建训练、验证和测试数据集
    train_dataset = CPDataset(
        args.dataroot, args.resolution, mode="train", data_list=args.train_data_list
    )
    validation_dataset = CPDataset(
        args.dataroot,
        args.resolution,
        mode="train",
        data_list=args.validation_data_list,
    )
    test_dataset = CPDataset(
        args.dataroot, args.resolution, mode="test", data_list=args.test_data_list
    )

    # 创建训练、验证和测试数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,  # 在每个epoch混洗数据
        batch_size=args.train_batch_size,  # 设置批处理大小
        num_workers=args.dataloader_num_workers,  # 设置数据加载的工作进程数
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=True,  # 在每个epoch混洗数据
        batch_size=args.train_batch_size,  # 设置批处理大小
        num_workers=args.dataloader_num_workers,  # 设置数据加载的工作进程数
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,  # 在每个epoch混洗数据
        batch_size=args.train_batch_size,  # 设置批处理大小
        num_workers=args.dataloader_num_workers,  # 设置数据加载的工作进程数
    )

    # Scheduler and math around the number of training steps.
    # 定义一个标志变量，用来记录是否重写了最大训练步数
    overrode_max_train_steps = False
    # 计算每个epoch中更新步数的数量
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    # 如果没有指定最大训练步数，则根据epoch数和每个epoch的更新步数来计算
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # 初始化学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 将模型的各个部分设置为不需要计算梯度
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.image_encoder.requires_grad_(False)

    # 根据配置，决定是否为对应的模型启用梯度检查点
    if args.gradient_checkpointing_vton:
        model.unet_vton.train()
        model.unet_vton.enable_gradient_checkpointing()
    else:
        model.unet_vton.requires_grad_(False)

    if args.gradient_checkpointing_garm:
        model.unet_garm.train()
        model.unet_garm.enable_gradient_checkpointing()
    else:
        model.unet_garm.requires_grad_(False)

    # Prepare everything with our `accelerator`.
    # 使用accelerator准备模型、优化器、数据加载器和学习率调度器
    (
        model.unet_garm,
        model.vae,
        model.unet_vton,
        optimizer,
        train_dataloader,
        test_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model.unet_garm,
        model.vae,
        model.unet_vton,
        optimizer,
        train_dataloader,
        test_dataloader,
        lr_scheduler,
    )

    # For mixed precision training we cast untrained weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # 根据是否使用混合精度训练，设置模型权重的数据类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 将模型的各个部分转移到适当的设备上，并根据需要设置数据类型
    model.vae.to(accelerator.device, dtype=weight_dtype)
    model.unet_garm.to(accelerator.device)
    model.unet_vton.to(accelerator.device)
    model.text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # 重新计算总训练步数，因为训练数据加载器的大小可能已经改变
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    # 重新计算总训练周期数
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # 如果在主进程上，初始化跟踪器并存储配置
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"entity": args.tracker_entity}},
        )

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    # 输出训练相关的信息
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # 根据是否有继续训练的需求，加载之前的训练状态
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            # 获取最新的检查点
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # 初始化进度条来显示训练进度
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.  在每台机器上只显示一次进度条
        disable=not accelerator.is_local_main_process,
    )

    # 初始化用于记录梯度的字典
    image_logs = None
    unet_garm_grad_dict = defaultdict(list)
    unet_vton_grad_dict = defaultdict(list)
    vae_grad_dict = defaultdict(list)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    # 解包模型，以便在使用 torch.compile编译的模型中正常使用
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # pre-train validation
    # 预训练验证
    log_validation(
        model, args, accelerator, weight_dtype, test_dataloader, validation_dataloader
    )

    # training starts!
    # 训练开始!
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # with accelerator.accumulate(model.unet_garm):
            # 使用自动混合精度来加速训练，同时减少内存消耗
            with torch.autocast("cuda"):
                image_garm = batch["ref_imgs"]  # 参考图像
                image_vton = batch["inpaint_image"]  # 待修复图像
                image_ori = batch["GT"]  # 原始图像
                inpaint_mask = batch["inpaint_mask"]  # 修复掩码
                mask = batch[
                    "mask"
                ]  # mask will not be used in trainning  训练时不使用的掩码
                prompt = batch["prompt"]  # 描述信息

                # 处理输入图像和文本提示
                prompt_image = model.auto_processor(
                    images=image_garm, return_tensors="pt"
                ).to(accelerator.device)
                prompt_image = model.image_encoder(
                    prompt_image.data["pixel_values"]
                ).image_embeds
                prompt_image = prompt_image.unsqueeze(1)
                prompt_embeds = model.text_encoder(
                    model.tokenize_captions(prompt, 2).to(accelerator.device)
                )[0]

                prompt_embeds[:, 1:] = prompt_image[:]

                # 编码提示信息
                prompt_embeds = model._encode_prompt(
                    prompt=prompt,
                    device=accelerator.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    prompt_embeds=prompt_embeds,
                )

                # 预处理图像
                image_garm = model.image_processor.preprocess(image_garm)
                image_vton = model.image_processor.preprocess(image_vton)
                image_ori = model.image_processor.preprocess(image_ori)
                mask = mask.unsqueeze(dim=1)

                # Convert images to latent space
                # 将图像转换为潜在空间表示
                garm_latents = model.prepare_garm_latents(
                    image=image_garm,
                    batch_size=args.train_batch_size,
                    num_images_per_prompt=1,
                    dtype=prompt_embeds.dtype,
                    device=accelerator.device,
                    do_classifier_free_guidance=False,
                )

                # 准备虚拟试穿的潜在表示和掩码
                vton_latents, mask_latents, ori_latents = model.prepare_vton_latents(
                    image=image_vton,
                    mask=mask,
                    image_ori=image_ori,
                    batch_size=args.train_batch_size,
                    num_images_per_prompt=1,
                    dtype=prompt_embeds.dtype,
                    device=accelerator.device,
                    do_classifier_free_guidance=False,
                )

                # TODO: why do we need to use sample() instead of mode()
                # 使用VAE编码并采样潜在空间
                latents = model.vae.encode(image_ori).latent_dist.sample()
                # latents = vae.encode(image_ori.to(weight_dtype).latent_dist.sample().to(accelerator.device))
                latents = latents * model.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                # 为潜在空间的噪声添加随机噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                # 对每个图像随机选择一个时间步
                timesteps = torch.randint(
                    0,
                    model.scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=accelerator.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # 根据每个时间步的噪声幅度添加噪声到潜在表示中（这是前向扩散过程）
                noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)

                # 使用UNet处理衣物图像的潜在表示，并计算空间注意力输出
                _, spatial_attn_outputs = model.unet_garm(
                    garm_latents,
                    0,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )

                # 将噪声潜在表示和虚拟试穿潜在表示结合
                latent_vton_model_input = torch.cat(
                    [noisy_latents, vton_latents], dim=1
                )
                spatial_attn_inputs = spatial_attn_outputs.copy()

                # 使用虚拟试穿UNet，根据输入的潜在表示和空间注意力输出预测去噪声结果
                noise_pred = model.unet_vton(
                    latent_vton_model_input,
                    spatial_attn_inputs,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                # 计算 MSE 损失函数，使用平均值作为缩减方法
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                # TODO: Are these latents x0 or xt-1?
                # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
                # extra_step_kwargs = model.prepare_extra_step_kwargs(generator, args.eta)
                # compute the previous noisy sample x_t -> x_t-1
                # latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # 使用 Accelerator 来管理模型的反向传播
                accelerator.backward(loss)
                # TODO: Do we need to clip gradients?
                # 判断是否需要同步梯度
                if accelerator.sync_gradients:
                    # 判断是否需要裁剪梯度
                    if args.clip_grad_norm:
                        # 对 unet_garm 模型的梯度进行裁剪
                        accelerator.clip_grad_norm_(
                            model.unet_garm.parameters(), args.max_grad_norm
                        )
                        # accelerator.clip_grad_norm_(model.unet_vton.parameters(), args.max_grad_norm)
                        # accelerator.clip_grad_norm_(model.vae.parameters(), args.max_grad_norm)

                # 判断是否需要记录梯度信息
                if args.log_grads:
                    # 如果 unet_garm 模型处于训练状态
                    if model.unet_garm.training:
                        # 遍历模型的每个子模块
                        for name, block in model.unet_garm.module.named_children():
                            grad = torch.tensor(0.0).to(accelerator.device)
                            # 计算所有参数的梯度范数之和
                            for p in block.parameters():
                                if p.grad is not None:
                                    grad += p.grad.norm()
                                    # grad += p.grad.abs().max()
                            # 记录每个子模块的梯度范数
                            unet_garm_grad_dict[name + ".grad.norm"] = (
                                grad.detach().item()
                            )
                        # 使用 Accelerator 记录梯度信息
                        accelerator.log(unet_garm_grad_dict, step=global_step)

                    if model.unet_vton.training:
                        for name, block in model.unet_vton.module.named_children():
                            grad = torch.tensor(0.0).to(accelerator.device)
                            for p in block.parameters():
                                if p.grad is not None:
                                    grad += p.grad.norm()
                            unet_vton_grad_dict[name + ".grad"] = grad.detach().item()
                        accelerator.log(unet_vton_grad_dict, step=global_step)

                    if model.vae.training:
                        for name, block in model.vae.named_children():
                            grad = torch.tensor(0.0).to(accelerator.device)
                            for p in block.parameters():
                                if p.grad is not None:
                                    grad += p.grad.norm()
                            vae_grad_dict[name + ".grad"] = grad.detach().item()
                        accelerator.log(vae_grad_dict, step=global_step)

                # 执行优化器的步骤，更新模型参数
                optimizer.step()
                # 更新学习率
                lr_scheduler.step()
                # 清除梯度
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            # 检查是否完成了一个优化步骤
            if accelerator.sync_gradients:
                # 更新进度条
                progress_bar.update(1)
                # 增加全局步骤计数
                global_step += 1

                # 如果是主进程
                if accelerator.is_main_process:
                    # Save the checkpoint
                    # 每隔一定步骤保存模型检查点
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        # 如果设置了检查点总数限制
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            # 确保不超过检查点数限制
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        # 保存当前模型状态
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        unet_vton = unwrap_model(model.unet_vton)
                        unet_vton.save_pretrained(
                            f"{args.output_dir}/unet_vton/checkpoint-{global_step}",
                            safe_serialization=True,
                        )

                        unet_garm = unwrap_model(model.unet_vton)
                        unet_garm.save_pretrained(
                            f"{args.output_dir}/unet_garm/checkpoint-{global_step}",
                            safe_serialization=True,
                        )

                    if global_step % args.validation_steps == 0:
                        log_validation(
                            model,
                            args,
                            accelerator,
                            weight_dtype,
                            test_dataloader,
                            validation_dataloader,
                        )

            # 记录损失和学习率
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            # 更新进度条
            progress_bar.set_postfix(**logs)
            # 使用 Accelerator 记录日志
            accelerator.log(logs, step=global_step)

            # 如果达到最大训练步数，结束训练
            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    # 等待所有加速器准备就绪
    accelerator.wait_for_everyone()
    # 如果当前进程是主进程
    if accelerator.is_main_process:
        # 解包模型中的 unet_vton 部分
        unet_vton = unwrap_model(model.unet_vton)
        # 将unet_vton模型保存到指定的输出目录，并开启安全序列化
        unet_vton.save_pretrained(
            args.output_dir + "/unet_vton", safe_serialization=True
        )

        # 解包模型中的 unet_garm 部分（这里应该是一个错误，应该是 model.unet_garm）
        unet_garm = unwrap_model(
            model.unet_vton
        )  # 注意这里可能是一个错误，应该是 model.unet_garm
        # 将unet_garm模型保存到指定的输出目录，并开启安全序列化
        unet_garm.save_pretrained(
            args.output_dir + "/unet_garm", safe_serialization=True
        )

        # Run a final round of validation.
        # 初始化image_logs变量
        image_logs = None
        # 如果定义了验证提示，则执行验证并记录图像
        if args.validation_prompt is not None:
            image_logs = log_validation(
                model,
                args,
                accelerator,
                weight_dtype,
                test_dataloader,
                validation_dataloader,
            )

        # 如果设置了推送到仓库的标志
        if args.push_to_hub:
            # 保存模型的信息卡片
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            # 将输出目录上传到指定的仓库，提交时忽略某些文件
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    # 结束训练，清理加速器资源
    accelerator.end_training()


# 如果这个文件作为主程序运行
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 执行主函数
    main(args)
