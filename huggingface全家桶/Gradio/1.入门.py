import gradio as gr
import cv2


def greet(name):
    return "Hello " + name + "!"


# interface = gr.Interface(fn=greet, inputs="text", outputs="text")

# interface = gr.Interface(
#     fn=greet,
#     inputs=gr.Textbox(lines=5, placeholder="Enter your name here...", label="name"),
#     outputs=gr.Textbox(label="Greeting"),
# )


def turn_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


# interface = gr.Interface(fn=turn_gray, inputs=gr.Image(), outputs="image")


def file_path(path, name):
    return path


interface = gr.Interface(
    fn=file_path,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Radio(["mali", "xiaoming"]),
    ],
    outputs="text",
)

interface.launch()
