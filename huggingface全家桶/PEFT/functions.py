def get_loader(text_lens=100):
    import torch
    import random
    from transformers import BertTokenizer
    from datasets import Dataset

    # 初始化BERT分词器，使用自定义的词汇表文件，并设置模型的最大长度为512
    tokenizer = BertTokenizer(vocab_file="tokenizer/vocab.txt", model_max_length=512)

    # 定义一个生成器函数，用于生成数据
    def f():
        for _ in range(2000):
            # 随机生成一个0到9之间的整数作为标签
            label = random.randint(0, 9)
            # 将标签重复text_lens次，生成文本
            text = " ".join(str(label) * text_lens)
            yield {"text": text, "label": label}

    # 使用生成器函数创建数据集
    dataset = Dataset.from_generator(f)

    # 定义一个函数，用于处理数据集中的每个样本
    def f(data):
        text = [i["text"] for i in data]
        label = [i["label"] for i in data]

        # 使用分词器对文本进行分词，并进行填充和截断
        data = tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # 将标签转换为torch.LongTensor
        data["labels"] = torch.LongTensor(label)

        return data

    # 创建数据加载器，使用自定义的处理函数，设置批量大小为32，随机打乱数据，丢弃最后一个不足批量大小的数据
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=f
    )

    return tokenizer, dataset, loader


def get_model(num_hidden_layers=32):
    import torch
    from transformers import BertConfig, BertForSequenceClassification
    from transformers.optimization import get_scheduler

    # 初始化BERT配置，设置标签数量为10，隐藏层数量为num_hidden_layers
    config = BertConfig(num_labels=10, num_hidden_layers=num_hidden_layers)
    # 使用配置初始化BERT模型
    model = BertForSequenceClassification(config)

    # 初始化Adam优化器，学习率为1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 初始化学习率调度器，使用余弦衰减，没有预热步骤，总训练步骤为50
    scheduler = get_scheduler(
        name="cosine", num_warmup_steps=0, num_training_steps=50, optimizer=optimizer
    )

    return model, optimizer, scheduler
