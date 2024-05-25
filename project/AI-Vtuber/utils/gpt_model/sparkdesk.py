import logging

from sparkdesk_web.core import SparkWeb
from sparkdesk_api.core import SparkAPI

from utils.common import Common
from utils.logger import Configure_logger


class SPARKDESK:
    def __init__(self, data):
        self.common = Common()
        # 日志文件路径
        file_path = "./log/log-" + self.common.get_bj_time(1) + ".txt"
        Configure_logger(file_path)

        self.type = data["type"]


        self.sparkWeb = None
        self.sparkAPI = None

        if data["cookie"] != "" and data["fd"] != "" and data["GtToken"] != "":
            self.sparkWeb = SparkWeb(
                cookie = data["cookie"],
                fd = data["fd"],
                GtToken = data["GtToken"]
            )
        elif data["app_id"] != "" and data["api_secret"] != "" and data["api_key"] != "":
            if data["assistant_id"] == "":
                self.sparkAPI = SparkAPI(
                    app_id = data["app_id"],
                    api_secret = data["api_secret"],
                    api_key = data["api_key"],
                    version = data["version"]
                )
            else:
                try:
                    self.sparkAPI = SparkAPI(
                        app_id = data["app_id"],
                        api_secret = data["api_secret"],
                        api_key = data["api_key"],
                        version = data["version"],
                        assistant_id = data["assistant_id"]
                    )
                except TypeError as e:
                    logging.error(e)
                    logging.error("如果没有assistant_id传参，说明你的sparkdesk-api库版本太低，请更新至最新版本。\n请先激活conda环境，然后更新，参考命令：pip install git+https://gitee.com/ikaros-521/sparkdesk-api -U")
        else:
            logging.info("讯飞星火配置为空")


    def get_resp(self, prompt):
        if self.type == "web":
            return self.sparkWeb.chat(prompt)
        elif self.type == "api":
            return self.sparkAPI.chat(prompt)
        else:
            logging.error("你瞎动什么配置？？？")
            exit(0)
