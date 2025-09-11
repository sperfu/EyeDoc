# @Author       : Duhongkai
# @Time         : 2024/1/20 17:16
# @Description  : 对QA进行验证，不适用Trainer训练器，生成的句子和有重复，尚未知晓原因

import json
from transformers import AutoTokenizer

from utils import util


class Evaluate:
    def __init__(self, param):
        self.param = param
        self.print_param()
        self.tokenizer = self.load_tokenizer()

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(f"{self.param['base_model']}")
        tokenizer.pad_token_id = 0
        return tokenizer

    # 验证
    def evaluate(self):
        # 加载label
        with open(f"data/test_QA_{self.param['test_count']}.json", "r", encoding="utf-8") as file:
            test_data_json = [json.loads(single_data) for single_data in file.readlines()]
            questions, labels, disease_index = zip(
                *[(d['question'], d['answer'], d['disease_index']) for d in test_data_json])
            label_list = list()
            for label in labels:
                label_token = self.tokenizer(label, return_tensors=None)["input_ids"]
                label_list.append(label_token)
        # 加载predict
        with open(f"data/{self.param['output_text']}", "r", encoding="utf-8") as file:
            predict_lines = file.readlines()
            predict_label = list()
            for predict in predict_lines:
                output = self.tokenizer(predict, return_tensors=None)["input_ids"]
                predict_label.append(output)
        # 计算结果
        res = util.compute_metrics(label_list, predict_label)
        return res

    def print_param(self):
        print(self.param)


if __name__ == '__main__':
    bloom_3b_knowledge_param = {
        'test_count': 500,
        'output_text': 'gen_4_delete_repeat.txt',
        "standard_name": "lora_bloom_eyeQA",
        "base_model": "Llama-2-7b-chinese-chat",
        # "base_model": "bloom-zh-3b",
    }
    eva = Evaluate(bloom_3b_knowledge_param)
    # 结果预测
    eva.evaluate()
