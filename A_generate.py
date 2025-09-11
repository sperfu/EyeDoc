# @Author       : Duhongkai
# @Time         : 2024/1/15 16:16
# @Description  : QA模型生成，数据在验证之前需要先进行生成
from peft import PeftModel
from similarities import BertSimilarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import json
import random
import numpy as np

from utils.prompter import Prompter
from utils import util, dataset
import os


class Generate:
    def __init__(self, param):
        self.param = param
        self.prompter = Prompter(self.param['template'])
        # 仅执行一次即可
        self.questions, self.labels, self.disease_index = self.build_qa_test()
        self.tokenizer, self.model = self.load_model()

    def build_qa_test(self):
        # 如果QA_Test已经存在，直接加载即可
        if os.path.exists(f"data/test_QA_{self.param['test_count']}.json"):
            with open(f"data/test_QA_{self.param['test_count']}.json", "r", encoding="utf-8") as file:
                test_data_json = [json.loads(single_data) for single_data in file.readlines()]
                questions, labels, disease_index = zip(
                    *[(d['question'], d['answer'], d['disease_index']) for d in test_data_json])
        else:
            with open(self.param["test_path"], 'r', encoding='utf-8') as file:
                all_data_json = [json.loads(single_data) for single_data in file.readlines()]
            data_sample = random.sample(range(0, len(all_data_json)), self.param["test_count"])
            all_data_np = np.array(all_data_json, )
            test_data_json = all_data_np[data_sample].tolist()
            questions, _, labels = zip(*[[d['instruction'], d['input'], d['output']] for d in test_data_json])
            disease_index = self.meet_test_disease(list(questions), list(labels))
        return questions, labels, disease_index

    def meet_test_disease(self, questions, labels):
        disease_model = BertSimilarity(model_name_or_path=self.param["disease_model"])
        disease_data = util.load_disease_data(self.param["knowledge_path"])
        disease_model.add_corpus(disease_data)
        res = disease_model.most_similar(queries=questions, topn=1)
        # 获取疾病知识
        disease_index = []
        with open(f"data/test_QA_{self.param['test_count']}.json", "w", encoding="utf-8") as file:
            for question, label, single_res in zip(questions, labels, res.values()):
                disease_index.append(list(single_res.keys())[0])
                data_json = json.dumps(
                    {"question": question, "answer": label, "disease_index": list(single_res.keys())[0]},
                    ensure_ascii=False)
                file.write(data_json + "\n")
        del disease_model, disease_data
        return disease_index

    def meet_disease(self, question):
        # 获取疾病知识
        knowledge_data = util.load_knowledge_data(self.param["knowledge_path"])
        # 判断最相近的疾病名称
        # disease_model = BertSimilarity(model_name_or_path=self.param["disease_model"])
        # disease_data = util.load_disease_data(self.param["knowledge_path"])
        # disease_model.add_corpus(disease_data)
        # res = disease_model.most_similar(queries=question, topn=1)
        # del disease_model, disease_data
        # return knowledge_data[list(res[0].keys())[0]]
        return knowledge_data[506]

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(f"{self.param['base_model']}")
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            self.param['base_model'],
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        eye_model = PeftModel.from_pretrained(
            model,
            self.param["lora_model"],
            torch_dtype=torch.float16,
        )
        eye_model.config.pad_token_id = 0  # unk
        eye_model.config.bos_token_id = 1
        eye_model.config.eos_token_id = 2
        eye_model.eval()
        return tokenizer, eye_model

    def init_dataset(self):
        knowledge_data = util.load_knowledge_data(self.param["knowledge_path"])
        test_input = list()
        for question, label, index in zip(self.questions, self.labels, self.disease_index):
            full_knowledge, full_qa = self.prompter.generate_prompt(
                knowledge=knowledge_data[index][:512],
                instruction=question,
            # )
            inputs = self.tokenizer(full_knowledge, full_qa, padding=False, return_tensors="pt")
            test_input.append(inputs["input_ids"].to("cuda"))
        return dataset.MyDataset(test_input)

    def generate(self, test_dataset):
        generation_config = GenerationConfig(
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            num_beams=4,
            repetition_penalty=2
        )
        for batch in tqdm(test_dataset):
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=batch["input_ids"],
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=256,
                    logits_processor=None,
                )
                s = generation_output.sequences[0]
                output = self.tokenizer.decode(s)
                output = self.prompter.get_response(output)
                with open(f"data/{self.param['output_text']}", "a", encoding="utf-8") as file:
                    file.write(output + "\n")

    def generate_qa(self, question, remove_repeat=True):
        knowlege_data = self.meet_disease(question)[:450]
        raw_response = util.evaluate_knowledge(knowlege_data, question, self.model, self.prompter, self.tokenizer)
        sentence = raw_response
        if remove_repeat:
            sentence = util.remove_repeat(raw_response)
        return raw_response, sentence


if __name__ == '__main__':
    llama_7b_param = {
        "standard_name": "lora_bloom_eyeQA",
        "test_count": "500",  # 测试集的数量，如果测试集不存在，会生成该测试集。
        "base_model": "Llama-2-7b-chinese-chat",  # the only required argument
        "lora_model": "lora/llama-eye-7b",
        "output_text": "gen_4.txt",
        "template": {
            "prompt_input": "下面是一个问题，运用医学知识来正确回答提问.\n### 问题:\n{instruction}\n### 回答:\n",
            "response_split": "### 回答:"
        }
    }
    bloom_3b_param = {
        "standard_name": "lora_bloom_eyeQA",
        "test_count": "500",  # 测试集的数量，如果测试集不存在，会生成该测试集。
        "base_model": "bloom-zh-3b",
        "lora_model": "lora/bloom-eye-3b",
        "output_text": "gen_3.txt",
        "template": {
            "prompt_input": "下面是一个问题，运用医学知识来正确回答提问.\n### 问题:\n{instruction}\n### 回答:\n",
            "response_split": "### 回答:"
        }
    }
    bloom_3b_knowledge_param = {
        "base_model": "bloom-zh-3b",
        "test_count": "500",  # 测试集的数量，如果测试集不存在，会生成该测试集。
        "lora_model": "lora/bloom-eye-3b-knowledge/checkpoint-5408",
        "knowledge_path": "data/eye_disease_knowledge_processed.json",
        "disease_model": "text2vec-similar",
        "output_text": "gen_2.txt",
        "template": {
            "prompt_input": "下面是一个眼部疾病相关的问题，请运用医学知识来正确回答提问。这里提供了一些可以参考的消息。"
                            "\n### 参考信息:\n{knowledge}"
                            "\n### 问题:\n{instruction}"
                            "\n### 回答:\n",
            "response_split": "### 回答:"
        }
    }

    generate = Generate(bloom_3b_knowledge_param)

    # 生成测试集
    # test_dataset = generate.init_dataset()
    # generate.generate(test_dataset=test_dataset)

    # 单句测试
    # question = "现在发现拍照眼睛一个大一个小，左眼无神，看上去空洞的，去看过眼科专家，检查不出来什么问题，滴了治干眼症的眼药水得到好转了，左眼也正常了，但是拍照看着总是很奇怪"
    # question = "眼睛弱视会导致失明吗"
    question = "眼睛视力不好有点疲劳，是什么原因？"
    raw_response, response = generate.generate_qa(question=question, remove_repeat=True)
    print(response)
