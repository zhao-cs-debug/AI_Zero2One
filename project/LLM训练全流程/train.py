import os
import argparse
from tqdm import tqdm
from os.path import join
from loguru import logger
from itertools import chain
import torch
import torch.nn as nn
import datasets
from datasets import load_dataset, concatenate_datasets

from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import bitsandbytes as bnb
from trl import DPOTrainer, get_kbit_device_map

from core.collator import PretrainCollator, SFTDataCollator
from core.argument import CustomArguments
from core.template import template_dict
from core.dataset import SFTDataset, DPODataset

from unsloth import FastLanguageModel

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, help="")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_config = args.train_config
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    custom_args, training_args = parser.parse_json_file(json_file=train_config)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    set_seed(training_args.seed)
    return custom_args, training_args

def load_tokenizer_model(custom_args, training_args):
    logger.info("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(custom_args.model_name_or_path, trust_remote_code=True, use_fast=True)

    logger.info("Initializing Model...")
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    if custom_args.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
    )
    model = AutoModelForCausalLM.from_pretrained(custom_args.model_name_or_path, **model_kwargs)

    if custom_args.section == 'pretrain':
        ref_model = None
        if custom_args.train_mode == 'full':
            peft_config = None
        else:
            raise Exception("Without considering the inclusion of LoRA in the pre-training process")

    if custom_args.section == 'sft':
        ref_model = None
        if custom_args.train_mode == 'full':
            peft_config = None
        else:
            if custom_args.train_mode == 'lora' and training_args.gradient_checkpointing:
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                linear_class = nn.Linear
            if custom_args.train_mode == 'qlora':
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
                linear_class = bnb.nn.Linear4bit
            target_modules = set()
            for name, module in model.named_modules():
                if isinstance(module, linear_class):
                    names = name.split('.')
                    target_modules.add(name[0] if len(names) == 1 else names[-1])
            target_modules = list(target_modules.remove('lm_head') if 'lm_head' in target_modules else target_modules)
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=custom_args.lora_rank,
                target_modules=target_modules,
                lora_alpha=custom_args.lora_alpha,
                lora_dropout=custom_args.lora_dropout,
            )
            model = get_peft_model(model, peft_config=peft_config)
            model.print_trainable_parameters()

    if custom_args.section == 'dpo':
        ref_model = AutoModelForCausalLM.from_pretrained(custom_args.model_name_or_path, **model_kwargs) if custom_args.train_mode == 'full' else None
        if custom_args.train_mode == 'full':
            peft_config = None
        else:
            target_modules = set()
            linear_class = bnb.nn.Linear4bit if custom_args.train_mode == 'qlora' else nn.Linear
            for name, module in model.named_modules():
                if isinstance(module, linear_class):
                    names = name.split('.')
                    target_modules.add(name[0] if len(names) == 1 else names[-1])
            target_modules = list(target_modules.remove('lm_head') if 'lm_head' in target_modules else target_modules)
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=custom_args.lora_rank,
                target_modules=target_modules,
                lora_alpha=custom_args.lora_alpha,
                lora_dropout=custom_args.lora_dropout,
            )

    total = sum(params.numel() for params in model.parameters())
    logger.info("Total params: %.3fM" % (total / 1e6))
    return tokenizer, model, ref_model, peft_config

# 加速训练和减少显存，方法地址：https://github.com/unslothai/unsloth
def load_unsloth_model(custom_args, training_args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=custom_args.model_name_or_path,
        max_seq_length=custom_args.max_seq_length,
        load_in_4bit=True if custom_args.train_mode == 'qlora' else False,
        trust_remote_code=True,
    )
    if custom_args.train_mode in ['lora', 'qlora']:
        target_modules = set()
        linear_class = bnb.nn.Linear4bit if custom_args.train_mode == 'qlora' else nn.Linear
        for name, module in model.named_modules():
            if isinstance(module, linear_class):
                names = name.split('.')
                target_modules.add(name[0] if len(names) == 1 else names[-1])
        target_modules = list(target_modules.remove('lm_head') if 'lm_head' in target_modules else target_modules)
        model = FastLanguageModel.get_peft_model(
            model,
            r=custom_args.lora_rank,
            target_modules=target_modules,
            lora_alpha=custom_args.lora_alpha,
            lora_dropout=custom_args.lora_dropout,
            random_state=training_args.seed,
            max_seq_length=custom_args.max_seq_length,
        )
    return tokenizer, model, None, None

def load_pretrain_dataset(training_args, custom_args, tokenizer):
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        output = {'input_ids': output.input_ids}
        return output

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_path = custom_args.data
    max_seq_length = custom_args.max_seq_length
    cache_dir = join(data_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    logger.info('Pretraining data path: {}'.format(data_path))

    logger.info('Scanning all the training file...')
    files = []
    for root, dir_names, file_names in os.walk(data_path):
        for file_name in file_names:
            file = join(root, file_name)
            if file_name.endswith('.jsonl'):
                files.append(file)
    logger.info(f'Total num of training file: {len(files)}')

    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        pretrain_dataset = []
        for idx, file in enumerate(tqdm(files)):
            logger.info(f'Loading file: {file}')
            file_name = os.path.basename(file)
            file_name = file_name.replace('.jsonl', '')
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_path, exist_ok=True)

            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'Finished loading datasets-{file_name} from cache')
            except Exception:
                tmp_cache_path = join(cache_path, 'tmp')
                logger.info(f'There is no cache of file {file_name}, start preprocessing...')
                raw_dataset = load_dataset("json", data_files=file, cache_dir=tmp_cache_path, keep_in_memory=False)
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=custom_args.tokenize_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=custom_args.tokenize_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
                # 删除临时目录
                # shutil.rmtree(tmp_cache_path)

            logger.info(f"Training number of {file_name}: {len(processed_dataset['train'])}")
            if idx == 0:
                pretrain_dataset = processed_dataset['train']
            else:
                assert pretrain_dataset.features.type == processed_dataset["train"].features.type
                pretrain_dataset = concatenate_datasets([pretrain_dataset, processed_dataset["train"]])
    logger.info(f"Total training number: {len(pretrain_dataset)}")
    return pretrain_dataset

def initialize_core(custom_args, training_args):
    logger.info("Initializing core...")
    if custom_args.use_unsloth:
        tokenizer, model, ref_model, peft_config = load_unsloth_model(custom_args, training_args)
    else:
        tokenizer, model, ref_model, peft_config = load_tokenizer_model(custom_args, training_args)

    if custom_args.section == 'pretrain':
        logger.info('Section Pretrain...')
        train_dataset = load_pretrain_dataset(training_args, custom_args, tokenizer)
        data_collator = PretrainCollator(tokenizer=tokenizer, max_seq_length=custom_args.max_seq_length)
    elif custom_args.section == 'sft':
        logger.info('Section Supervised Fine-Tuning...')
        if custom_args.template_name not in template_dict.keys():
            raise Exception(f"The template does not exist; all available templates are displayed in：{template_dict.keys()}")
        template = template_dict[custom_args.template_name]
        logger.info('Loading Supervised Fine-Tuning Dataset...')
        train_dataset = SFTDataset(custom_args.data, tokenizer, custom_args.max_seq_length, template)
        data_collator = SFTDataCollator(tokenizer=tokenizer, max_seq_length=custom_args.max_seq_length)
    else:
        logger.info('Section Direct Preference Optimization...')
        if custom_args.template_name not in template_dict.keys():
            raise Exception(f"The template does not exist; all available templates are displayed in：{template_dict.keys()}")
        template = template_dict[custom_args.template_name]
        logger.info('Loading DPO Dataset...')
        train_dataset = DPODataset(custom_args.data, tokenizer, custom_args.max_seq_length, custom_args.max_prompt_length, template)
        data_collator = None

    if custom_args.section == 'pretrain' or custom_args.section == 'sft':
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
    else:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            beta=custom_args.beta,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    return trainer

def main():
    custom_args, training_args = create_args()
    trainer = initialize_core(custom_args, training_args)
    logger.info("Starting Training...")
    result = trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    metrics = result.metrics    # 训练指标
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

if __name__ == "__main__":
    main()

# deepspeed --num_gpus=1 train.py --train_config config/pretrain/qwen1.5-7b-pretrain-full.json
# python train.py --train_config config/sft/qwen1.5-7b-sft-qlora.json
# python train.py --train_config config/dpo/qwen1.5-7b-dpo-qlora.json
