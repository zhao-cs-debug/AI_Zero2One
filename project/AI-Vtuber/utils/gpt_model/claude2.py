from curl_cffi import requests
import json
import os
import uuid
import re
import logging
import traceback

from utils.common import Common
from utils.logger import Configure_logger


class Claude2:

    def __init__(self, data):
        self.common = Common()
        # 日志文件路径
        file_path = "./log/log-" + self.common.get_bj_time(1) + ".txt"
        Configure_logger(file_path)

        try:
            self.cookie = data["cookie"]
            self.use_proxy = data["use_proxy"]
            if self.use_proxy:
                self.proxies = data["proxies"]
            else:
                self.proxies = None
            self.organization_id = None
            #self.organization_id ="28912dc3-bcd3-43c5-944c-a943a02d19fc"

            if self.get_organization_id() is None:
                logging.error("获取organization_id失败！Claude2将无法正常工作！请排查问题")

            self.conversation_id = self.create_new_chat()['uuid']
        except Exception as e:
            logging.error(traceback.format_exc())


    def get_organization_id(self):
        url = "https://claude.ai/api/organizations"

        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.125 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://claude.ai/chats',
            'Content-Type': 'application/json',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Connection': 'keep-alive',
            'Cookie': f'{self.cookie}'
        }

        response = self.send_request("GET",url,headers=headers)
        if response.status_code == 200:
            res = json.loads(response.text)
            uuid = res[0]['uuid']

            self.organization_id = uuid

            logging.info(f"创建新会话：{uuid}")

            return uuid
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")
            return None

    def get_content_type(self, file_path):
        # Function to determine content type based on file extension
        extension = os.path.splitext(file_path)[-1].lower()
        if extension == '.pdf':
            return 'application/pdf'
        elif extension == '.txt':
            return 'text/plain'
        elif extension == '.csv':
            return 'text/csv'
        # Add more content types as needed for other file types
        else:
            return 'application/octet-stream'

    # Lists all the conversations you had with Claude
    def list_all_conversations(self):
        url = f"https://claude.ai/api/organizations/{self.organization_id}/chat_conversations"

        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://claude.ai/chats',
            'Content-Type': 'application/json',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Connection': 'keep-alive',
            'Cookie': f'{self.cookie}'
        }

        response = self.send_request("GET",url,headers=headers)
        conversations = response.json()

        # Returns all conversation information in a list
        if response.status_code == 200:
            return conversations
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")

    # Send Message to Claude
    def send_message(self, prompt, conversation_id, attachment=None):
        url = "https://claude.ai/api/append_message"
        #print("send_message,attachment"+attachment)
        # Upload attachment if provided
        attachments = []
        if attachment:
            attachment_response = self.upload_attachment(attachment)
            if attachment_response:
                attachments = [attachment_response]
            else:
                logging.error("Error: Invalid file format. Please try again.")
                return None

        # Ensure attachments is an empty list when no attachment is provided
        if not attachment:
            attachments = []

        payload = json.dumps({
            "completion": {
                "prompt": f"{prompt}",
                "timezone": "Asia/Kolkata",
                "model": "claude-2.1"
            },
            "organization_uuid": f"{self.organization_id}",
            "conversation_uuid": f"{conversation_id}",
            "text": f"{prompt}",
            "attachments": attachments
        })

        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Accept': 'text/event-stream, text/event-stream',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://claude.ai/chats',
            'Content-Type': 'application/json',
            'Origin': 'https://claude.ai',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Cookie': f'{self.cookie}',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'TE': 'trailers'
        }

        response = self.send_request("POST",url,headers=headers, data=payload, stream=True)
        decoded_data = response.content.decode("utf-8")
        #logger.info("send_message {} decoded_data：".format(decoded_data))
        decoded_data = re.sub('\n+', '\n', decoded_data).strip()
        data_strings = decoded_data.split('\n')
        completions = []
        for data_string in data_strings:
            json_str = data_string[6:].strip()
            data = json.loads(json_str)
            if 'completion' in data:
                completions.append(data['completion'])

        answer = ''.join(completions)
        logging.debug("Claude2:{}".format(answer))
        return answer

    # Deletes the conversation
    def delete_conversation(self, conversation_id):
        url = f"https://claude.ai/api/organizations/{self.organization_id}/chat_conversations/{conversation_id}"

        payload = json.dumps(f"{conversation_id}")
        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Accept-Language': 'en-US,en;q=0.5',
            'Content-Type': 'application/json',
            'Content-Length': '38',
            'Referer': 'https://claude.ai/chats',
            'Origin': 'https://claude.ai',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Connection': 'keep-alive',
            'Cookie': f'{self.cookie}',
            'TE': 'trailers'
        }

        response = self.send_request("DELETE",url,headers=headers, data=payload)
        # Returns True if deleted or False if any error in deleting
        if response.status_code == 200:
            return True
        else:
            return False

    # Returns all the messages in conversation
    def chat_conversation_history(self, conversation_id):
        url = f"https://claude.ai/api/organizations/{self.organization_id}/chat_conversations/{conversation_id}"

        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://claude.ai/chats',
            'Content-Type': 'application/json',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Connection': 'keep-alive',
            'Cookie': f'{self.cookie}'
        }

        response = self.send_request("GET",url,headers=headers,params={'encoding': 'utf-8'})
        logging.info(type(response))

        # List all the conversations in JSON
        return response.json()

    def generate_uuid(self):
        random_uuid = uuid.uuid4()
        random_uuid_str = str(random_uuid)
        formatted_uuid = f"{random_uuid_str[0:8]}-{random_uuid_str[9:13]}-{random_uuid_str[14:18]}-{random_uuid_str[19:23]}-{random_uuid_str[24:]}"
        return formatted_uuid

    def create_new_chat(self):
        url = f"https://claude.ai/api/organizations/{self.organization_id}/chat_conversations"
        uuid = self.generate_uuid()

        payload = json.dumps({"uuid": uuid, "name": ""})
        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://claude.ai/chats',
            'Content-Type': 'application/json',
            'Origin': 'https://claude.ai',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Cookie': self.cookie,
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'TE': 'trailers'
        }
        response = self.send_request("POST",url,headers=headers, data=payload)
        # Returns JSON of the newly created conversation information
        return response.json()

    # Resets all the conversations
    def reset_all(self):
        conversations = self.list_all_conversations()

        for conversation in conversations:
            conversation_id = conversation['uuid']
            delete_id = self.delete_conversation(conversation_id)

        return True

    def upload_attachment(self, file_path):
        if file_path.endswith(('.txt', '.pdf', '.csv')):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            file_type = "text/plain"
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            return {
                "file_name": file_name,
                "file_type": file_type,
                "file_size": file_size,
                "extracted_content": file_content
            }

        url = 'https://claude.ai/api/convert_document'
        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://claude.ai/chats',
            'Origin': 'https://claude.ai',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Connection': 'keep-alive',
            'Cookie': f'{self.cookie}',
            'TE': 'trailers'
        }

        file_name = os.path.basename(file_path)
        content_type = self.get_content_type(file_path)
        files = {
            'file': (file_name, open(file_path, 'rb'), content_type),
            'orgUuid': (None, self.organization_id)
        }
        response = self.send_request(url, "POST",headers=headers, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            return False

    # Renames the chat conversation title
    def rename_chat(self, title, conversation_id):
        url = "https://claude.ai/api/rename_chat"

        payload = json.dumps({
            "organization_uuid": f"{self.organization_id}",
            "conversation_uuid": f"{conversation_id}",
            "title": f"{title}"
        })
        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Accept-Language': 'en-US,en;q=0.5',
            'Content-Type': 'application/json',
            'Referer': 'https://claude.ai/chats',
            'Origin': 'https://claude.ai',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Connection': 'keep-alive',
            'Cookie': f'{self.cookie}',
            'TE': 'trailers'
        }

        response = self.send_request("POST",url,headers=headers, data=payload)
        if response.status_code == 200:
            return True
        else:
            return False

    def send_request(self, method, url, headers, data=None, files=None, params=None, stream=False):
        if self.use_proxy:
            return requests.request(method, url, headers=headers, data=data, files=files, params=params,impersonate="chrome110",proxies=self.proxies,timeout=500)
        else:
            return requests.request(method, url, headers=headers, data=data, files=files, params=params,impersonate="chrome110",timeout=500)
    

    # 获取Claude2的请求结果，共用一个conversation_id，变向记忆功能
    def get_resp(self, prompt):
        try:
            resp_content = self.send_message(prompt, self.conversation_id)
            return resp_content
        except Exception as e:
            logging.error(traceback.format_exc())
            return None


if __name__ == '__main__':
    data = {
        "cookie": "",
        "use_proxy": True,
        "proxies": {
        "http": "http://127.0.0.1:10809",
        "https": "http://127.0.0.1:10809",
        "socks5": "socks://127.0.0.1:10808"
        }
    }
    claude_api = Claude2(data)
    new_chat = claude_api.create_new_chat()
    conversation_id = new_chat['uuid']
    print(conversation_id)

    prompt = "Hello, Claude!"
    response = claude_api.send_message(prompt, conversation_id)
    print(response)
