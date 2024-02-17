"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    def __init__(self, template: dict = ""):
        # 将knowledge与template分开，有利于切分
        prompt_input = template["prompt_input"]
        split_index = prompt_input.index("\n### 问题:\n")
        self.knowledge_template = prompt_input[:split_index]
        self.qa_template = prompt_input[split_index:]
        self.template = template

    def generate_prompt(
        self,
        knowledge: str,
        instruction: str,
        output: Union[None, str] = None,
    ) -> (str, str):
        # 判断是否存在知识源
        if knowledge and len(knowledge) > 0:
            knowledge = self.knowledge_template.format(knowledge=knowledge)
        else:
            knowledge = self.knowledge_template
        qa = self.qa_template.format(instruction=instruction)
        if output:
            qa = f"{qa}{output}"
        return knowledge, qa

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


# 多轮次提示构造
class MultiPrompter(object):
    def __init__(self, template: dict = ""):
        self.prompt = template["prompt_input"]
        self.response_split = template["response_split"]

    def generate_prompt(
        self,
        history: str,
        output: Union[None, str] = None,
    ) -> (str, str):
        prompt = self.prompt.format(history=history)

        if output:
            prompt = f"{prompt}{output}"
        return prompt

    # 最后一个轮次的对话
    def get_response(self, output: str) -> str:
        return output.split(self.response_split)[-1].strip()
