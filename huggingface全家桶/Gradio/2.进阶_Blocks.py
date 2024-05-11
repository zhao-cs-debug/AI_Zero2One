import gradio as gr

# gr.Blocks()/gr.Row()/gr.Column()/gr.Tab()/gr.Group()/gr.Accordion()

with gr.Blocks(css="style.css") as demo:
    with gr.Row():
        gr.Dropdown(["1", "2", "3", "4"], label="Stable Diffusion checkpoint", scale=3)
        button = gr.Button(value="刷新", elem_classes="btn", scale=1, min_width=1)
        gr.HTML("&nbsp;")
        gr.HTML("&nbsp;")
        gr.HTML("&nbsp;")
        gr.HTML("&nbsp;")
        gr.HTML("&nbsp;")
    with gr.Tab(label="txt2img"):
        with gr.Row():
            with gr.Column(scale=15):
                txt1 = gr.Textbox(lines=2, label="")
                txt2 = gr.Textbox(lines=2, label="")
            with gr.Column(scale=1, min_width=1):
                button1 = gr.Button(value="1", elem_classes="btn")
                button2 = gr.Button(value="2", elem_classes="btn")
                button3 = gr.Button(value="3", elem_classes="btn")
                button4 = gr.Button(value="4", elem_classes="btn")
            with gr.Column(scale=6):
                generate_button = gr.Button(
                    value="Generate", variant="primary", scale=1
                )
                with gr.Row():
                    dropdown = gr.Dropdown(["1", "2", "3", "4"], label="Style1")
                    dropdown2 = gr.Dropdown(["1", "2", "3", "4"], label="Style2")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    dropdown3 = gr.Dropdown(
                        ["1", "2", "3", "4"], label="Sampling method"
                    )
                    slider1 = gr.Slider(minimum=0, maximum=100, label="Sampling steps")
                checkboxgroup = gr.CheckboxGroup(
                    ["Restore faces", "Tiling", "Hires.fix"], label=""
                )
                with gr.Row():
                    slider2 = gr.Slider(minimum=0, maximum=100, label="Width")
                    slider3 = gr.Slider(minimum=0, maximum=100, label="Batch count")
                with gr.Row():
                    slider4 = gr.Slider(minimum=0, maximum=100, label="Height")
                    slider5 = gr.Slider(minimum=0, maximum=100, label="Batch size")
                slider6 = gr.Slider(minimum=0, maximum=100, label="CFG Scale")
                with gr.Row():
                    number1 = gr.Number(label="Seed", scale=5)
                    button5 = gr.Button(value="Randomize", min_width=1)
                    button6 = gr.Button(value="Reset", min_width=1)
                    checkbox1 = gr.Checkbox(label="Extra", min_width=1)
                dropdown4 = gr.Dropdown(["1", "2", "3", "4"], label="Script")
            with gr.Column():
                gallery = gr.Gallery(
                    [
                        "https://upload.wikimedia.org/wikipedia/commons/4/4d/Acinonyx_jubatus.jpg",
                        "https://nationalzoo.si.edu/sites/default/files/animals/cheetah-002.jpg",
                        "https://img.etimg.com/thumb/msid-71424179,width-650,imgsize-126507,,resizemode-4,quality-100/cheetah.jpg",
                        "https://www.sciencenews.org/wp-content/uploads/2020/06/060320_mt_cheetah_feat-1028x579.jpg",
                    ],
                    columns=2,
                )
                with gr.Row():
                    button7 = gr.Button(value="Save", min_width=1)
                    button8 = gr.Button(value="Save", min_width=1)
                    button9 = gr.Button(value="Zip", min_width=1)
                    button10 = gr.Button(value="Send to img2img", min_width=1)
                    button11 = gr.Button(value="Send to inpaint", min_width=1)
                    button12 = gr.Button(value="Send to extras", min_width=1)
                txt3 = gr.Textbox(lines=4, label="")
    with gr.Tab(label="img2img"):  # 标签页
        with gr.Row():
            with gr.Group():  # 组
                button13 = gr.Button(value="test1", min_width=1)
                button14 = gr.Button(value="test2", min_width=1)
            with gr.Accordion():  # 折叠面板
                with gr.Row():  # 行
                    button15 = gr.Button(value="test3", min_width=1)
                    button16 = gr.Button(value="test4", min_width=1)
                with gr.Column():  # 列
                    button17 = gr.Button(value="test5", min_width=1)
                    button18 = gr.Button(value="test6", min_width=1)

demo.launch()
