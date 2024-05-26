import json, logging, traceback, requests
from wenxinworkshop import LLMAPI, AppBuilderAPI, EmbeddingAPI, PromptTemplateAPI
from wenxinworkshop import Message, Messages, Texts

from utils.common import Common
from utils.logger import Configure_logger

# 前往官网：https://cloud.baidu.com/product/wenxinworkshop 申请服务获取

class My_WenXinWorkShop:
    def __init__(self, data):
        self.common = Common()
        # 日志文件路径
        file_path = "./log/log-" + self.common.get_bj_time(1) + ".txt"
        Configure_logger(file_path)

        self.config_data = data
        self.history = []

        self.my_bot = None

        self.conversation_id = None

        logging.debug(self.config_data)

        try:
            if self.config_data['type'] == "千帆大模型":
                model_url_map = {
                    "ERNIEBot": LLMAPI.ERNIEBot,
                    "ERNIEBot_turbo": LLMAPI.ERNIEBot_turbo,
                    "ERNIEBot_4_0": LLMAPI.ERNIEBot_4_0,
                    "ERNIE_SPEED_128K": LLMAPI.ERNIEBot_4_0,
                    "ERNIE_SPEED_8K": LLMAPI.ERNIE_SPEED_8K,
                    "ERNIE_LITE_8K_0308": LLMAPI.ERNIE_LITE_8K_0308,
                    "ERNIE_LITE_8K_0922": LLMAPI.ERNIE_LITE_8K_0922,
                    "ERNIE_TINY_8K": LLMAPI.ERNIE_TINY_8K,
                    "BLOOMZ_7B": LLMAPI.BLOOMZ_7B,
                    "LLAMA_2_7B": LLMAPI.LLAMA_2_7B,
                    "LLAMA_2_13B": LLMAPI.LLAMA_2_13B,
                    "LLAMA_2_70B": LLMAPI.LLAMA_2_70B,
                    "ERNIEBot_4_0": LLMAPI.ERNIEBot_4_0,
                    "QIANFAN_BLOOMZ_7B_COMPRESSED": LLMAPI.QIANFAN_BLOOMZ_7B_COMPRESSED,
                    "QIANFAN_CHINESE_LLAMA_2_7B": LLMAPI.QIANFAN_CHINESE_LLAMA_2_7B,
                    "CHATGLM2_6B_32K": LLMAPI.CHATGLM2_6B_32K,
                    "AQUILACHAT_7B": LLMAPI.AQUILACHAT_7B,
                    "ERNIE_BOT_8K": LLMAPI.ERNIE_BOT_8K,
                    "CODELLAMA_7B_INSTRUCT": LLMAPI.CODELLAMA_7B_INSTRUCT,
                    "XUANYUAN_70B_CHAT": LLMAPI.XUANYUAN_70B_CHAT,
                    "CHATLAW": LLMAPI.QIANFAN_BLOOMZ_7B_COMPRESSED,
                    "QIANFAN_BLOOMZ_7B_COMPRESSED": LLMAPI.CHATLAW,
                }

                selected_model = self.config_data["model"]
                if selected_model in model_url_map:
                    self.my_bot = LLMAPI(
                        api_key=self.config_data["api_key"],
                        secret_key=self.config_data["secret_key"],
                        url=model_url_map[selected_model]
                    )
            elif self.config_data['type'] == "AppBuilder":
                self.app_builder_get_conversation_id()
        except Exception as e:
            logging.error(traceback.format_exc())


    def app_builder_get_conversation_id(self):
        try:
            url = "https://qianfan.baidubce.com/v2/app/conversation"
        
            payload = json.dumps({"app_id": self.config_data["app_id"]})
            headers = {
                'Content-Type': 'application/json',
                'X-Appbuilder-Authorization': f'Bearer {self.config_data["app_token"]}'
            }

            logging.debug(f'payload={payload}\nheaders={headers}')
            
            response = requests.request("POST", url, headers=headers, data=payload)
            resp_json = json.loads(response.content)
            if "conversation_id" in resp_json:
                self.conversation_id = resp_json["conversation_id"]
                logging.info(f"获取会话ID成功，会话ID为：{self.conversation_id}")
            else:
                logging.error(f"获取会话ID失败，请检查app_id/app_token是否正确。错误信息：{resp_json}")

            return None
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"获取会话ID失败，请检查app_id/app_token是否正确。错误信息：{e}")
            return None


    def get_resp(self, prompt):
        """请求对应接口，获取返回值

        Args:
            prompt (str): 你的提问

        Returns:
            str: 返回的文本回答
        """
        try:
            resp_content = None

            if self.config_data['type'] == "千帆大模型":
                # create messages
                messages: Messages = []
                
                for history in self.history:
                    messages.append(Message(
                        role=history["role"],
                        content=history["content"]
                    ))

                messages.append(Message(
                    role='user',
                    content=prompt
                ))

                logging.info(f"self.history={self.history}")

                # get response from LLM API
                resp_content = self.my_bot(
                    messages=messages,
                    temperature=self.config_data["temperature"],
                    top_p=self.config_data["top_p"],
                    penalty_score=self.config_data["penalty_score"],
                    stream=None,
                    user_id=None,
                    chunk_size=512
                )

                # 启用历史就给我记住！
                if self.config_data["history_enable"]:
                    while True:
                        # 获取嵌套列表中所有字符串的字符数
                        total_chars = sum(len(item['content']) for item in self.history if 'content' in item)
                        # 如果大于限定最大历史数，就剔除第一个元素
                        if total_chars > self.config_data["history_max_len"]:
                            self.history.pop(0)
                            self.history.pop(0)
                        else:
                            # self.history.pop()
                            self.history.append({"role": "user", "content": prompt})
                            self.history.append({"role": "assistant", "content": resp_content})
                            break
            elif self.config_data['type'] == "AppBuilder":
                url = "https://qianfan.baidubce.com/v2/app/conversation/runs"
    
                payload = json.dumps({
                    "app_id": self.config_data["app_id"],
                    "query": prompt,
                    "stream": False,
                    "conversation_id": self.conversation_id
                })
                headers = {
                    'Content-Type': 'application/json',
                    'X-Appbuilder-Authorization': f'Bearer {self.config_data["app_token"]}'
                }
                
                response = requests.request("POST", url, headers=headers, data=payload)
                resp_json = json.loads(response.content)
                
                logging.debug(f"resp_json={resp_json}")

                if "content" in resp_json:
                    for data in resp_json["content"]:
                        if data["event_status"] == "done":
                            resp_content = data["outputs"]["text"]
                else:
                    logging.error(f"获取LLM返回失败。{resp_json}")
                    return None

            return resp_content
            
        except Exception as e:
            logging.error(e)

        return None

if __name__ == '__main__':
    # 配置日志输出格式
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别，可以根据需求调整
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data = {
        "model": "ERNIEBot",
        "app_id": "",
        "app_token": "",
        "api_key": "",
        "secret_key": "",
        "top_p": 0.8,
        "temperature": 0.9,
        "penalty_score": 1.0,
        "history_enable": True,
        "history_max_len": 300
    }

    # 实例化并调用
    my_wenxinworkshop = My_WenXinWorkShop(data)
    logging.info(my_wenxinworkshop.get_resp("你可以扮演猫娘吗，每句话后面加个喵"))
    logging.info(my_wenxinworkshop.get_resp("早上好"))
