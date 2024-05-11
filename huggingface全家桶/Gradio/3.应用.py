import gradio as gr
import json
import socket

def doChatbot(message, history):
    history = [[y, x] for x, y in history]
    data = {
        'query': message,
        'history': history
    }
    data = json.dumps(data)
    server_address = ('xxxxx.xxx', 12345)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(server_address)
        sock.sendall(data.encode('utf-8'))
        response = receive_all(sock)
        return json.loads(response.decode('UTF-8'))['generated_text']
    except socket.error as e:
        return str(e)

def receive_all(sock):
    BUFF_SIZE = 4096
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            break
    return data

def start_chatbot():
    gr.ChatInterface(
        fn=doChatbot,
        chatbot=gr.Chatbot(height=500, value=[["你好", "您好，我是医疗助手，我将尽力帮助您解决问题。"]]),
        textbox=gr.Textbox(placeholder="请输入您的问题", container=False, scale=7),
        title="AI医疗助手",
        theme="soft",
        examples=["我感觉头晕，怎么办？", "我感觉胸口闷，怎么办？", "我感觉胸口疼，怎么办？"],
        retry_btn=None,
        submit_btn="发送",
        undo_btn="删除前言",
        clear_btn="清空",
    ).queue().launch(server_port=7860)

if __name__ == '__main__':
    start_chatbot()