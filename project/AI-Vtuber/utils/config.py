import json  # 导入json模块，用于处理json格式的数据

class Config:  # 定义一个名为Config的类
    # 单例模式
    # _instance = None  # 用于存储类的唯一实例
    config = None  # 用于存储配置信息

    # def __new__(cls, *args, **kwargs):  # 重写__new__方法，实现单例模式
    #     if not cls._instance:  # 如果类的实例不存在
    #         cls._instance = super(Config, cls).__new__(cls)  # 创建类的实例
    #     return cls._instance  # 返回类的实例

    def __init__(self, config_file):  # 类的初始化方法
        if self.config is None:  # 如果配置信息不存在
            with open(config_file, 'r', encoding="utf-8") as f:  # 打开配置文件
                self.config = json.load(f)  # 将配置文件内容加载为json格式

    def __getitem__(self, key):  # 定义一个特殊方法，用于获取配置信息
        return self.config.get(key)  # 返回指定键对应的值

    def get(self, *keys):  # 定义一个方法，用于获取嵌套的配置信息
        result = self.config  # 初始化结果为配置信息
        for key in keys:  # 遍历所有键
            result = result.get(key, None)  # 获取指定键对应的值
            if result is None:  # 如果结果为None
                break  # 跳出循环
        return result  # 返回结果