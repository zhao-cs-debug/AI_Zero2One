import gradio as gr

# 常用输入类型
input_list = [
    gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio"),        # 上传音频文件或者录音
    gr.Checkbox(label="Checkbox"),                                                  # 复选框
    gr.ColorPicker(label="ColorPicker"),                                            # 颜色选择器
    gr.Dataframe(label="DataFrame"),                                                # 数据框
    gr.Dropdown(["a", "b", "c"], label="Dropdown"),                                 # 下拉框
    gr.File(label="File", type="filepath"),                                         # 上传文件
    gr.Image(sources=["webcam", "upload"], label="Image"),                          # 上传图片或者拍照
    gr.Number(label="Number"),                                                      # 数字输入框
    gr.Radio(["a", "b", "c"], label="Radio"),                                       # 单选框
    gr.Slider(minimum=0, maximum=10, label="Slider", step=1),                       # 滑动条
    gr.Textbox(label="Textbox", lines=3, max_lines=5, placeholder="Placeholder"),   # 文本框
    gr.TextArea(label="TextArea", lines=3, max_lines=5, placeholder="Placeholder"), # 文本域
    gr.Video(sources=["webcam", "upload"], label="Video"),                          # 上传视频或者录像
    gr.CheckboxGroup(["a", "b", "c"], label="CheckboxGroup"),                       # 多选框
]

output_list = [
    gr.Textbox(label="Audio outputs", lines=7),
    gr.Textbox(label="Checkbox outputs"),
    gr.Textbox(label="ColorPicker outputs"),
    gr.Textbox(label="DataFrame outputs"),
    gr.Textbox(label="Dropdown outputs"),
    gr.Textbox(label="File outputs"),
    gr.Textbox(label="Image outputs"),
    gr.Textbox(label="Number outputs"),
    gr.Textbox(label="Radio outputs"),
    gr.Textbox(label="Slider outputs"),
    gr.Textbox(label="Textbox outputs"),
    gr.Textbox(label="TextArea outputs"),
    gr.Textbox(label="Video outputs"),
    gr.Textbox(label="CheckboxGroup outputs"),
]


def input_and_output(*data):
    return data


interface = gr.Interface(
    fn=input_and_output,
    inputs=input_list,
    outputs=output_list,
    title="Input and Output",
    description="This is a test for all input and output types.",
    live=True,  # 是否实时更新
)
interface.launch()

# 常用输出类型
def audio_output(audio):
    hz = audio[0]
    data = audio[1]
    return audio
demo = gr.Interface(fn=audio_output, inputs=gr.Audio(type="numpy"), outputs="audio")
def audio_output(audio):
    return audio
demo = gr.Interface(fn=audio_output, inputs=gr.Audio(type="filepath"), outputs="audio")

import cv2
def turn_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
demo = gr.Interface(fn=turn_gray,inputs=gr.Image(),outputs="image")

import pandas as pd
simple = pd.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": [7, 8, 9],
})
demo = gr.Interface(fn=None, inputs=None, outputs=gr.BarPlot(simple, x="a", y="b", label="BarPlot"))

def process():
    cheetahs = [
        "https://upload.wikimedia.org/wikipedia/commons/4/4d/Acinonyx_jubatus.jpg",
        "https://nationalzoo.si.edu/sites/default/files/animals/cheetah-002.jpg",
        "https://img.etimg.com/thumb/msid-71424179,width-650,imgsize-126507,,resizemode-4,quality-100/cheetah.jpg",
        "https://www.sciencenews.org/wp-content/uploads/2020/06/060320_mt_cheetah_feat-1028x579.jpg",
    ]
    cheetahs = [(cheetah, f"Cheetah {i+1}") for i, cheetah in enumerate(cheetahs)]
    return cheetahs
demo = gr.Interface(fn=process, inputs=None, outputs=gr.Gallery(columns=2, label="Cheetahs"))

import matplotlib.pyplot as plt
import numpy as np
def fig_output(df):
    Fs = 8000
    f = 5
    sample = 10
    x = np.arange(sample)
    y = np.sin(2 * np.pi * f * x / Fs)
    plt.plot(x, y)
    return plt
demo = gr.Interface(fn=fig_output, inputs=None, outputs=gr.Plot())

json_sample = {
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": [7, 8, 9],
}
demo = gr.Interface(fn=None, inputs=None, outputs=gr.Json(json_sample, label="Json"))

demo = gr.Interface(fn=None, inputs=None, outputs=gr.HTML("<h1>Hello, World!</h1>"))

# 一般情况下，输入组件可以作为输出组件来使用。组件如同外衣，可以随意搭配。
# 详细查看Gradio官方文档：https://gradio.app/docs
