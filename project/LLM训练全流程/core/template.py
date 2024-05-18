from dataclasses import dataclass
from typing import Dict


@dataclass
class Template:
    template_name: str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str

template_dict: Dict[str, Template] = dict()

def register_template(
    template_name, system_format, user_format, assistant_format, system, stop_word=None
):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
    )

# 注册template
register_template(
    template_name="default",
    system_format="System: {content}\n\n",
    user_format="User: {content}\nAssistant: ",
    assistant_format="{content} {stop_token}",
    system=None,
    stop_word=None,
)

register_template(
    template_name="qwen",
    system_format="<|im_start|>system\n{content}<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    assistant_format="{content}<|im_end|>\n",
    system="You are a helpful assistant.",
    stop_word="<|im_end|>",
)
