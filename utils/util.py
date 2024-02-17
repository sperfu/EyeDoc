# @Author       : Duhongkai
# @Time         : 2024/1/10 14:49
# @Description  : 工具类

import os
import re
import torch
from peft import set_peft_model_state_dict
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, GenerationConfig, \
    LogitsProcessor, LogitsProcessorList
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR  # checkpoint
# from openai import OpenAI
import json

from evaluate.rouge import get_rouge_l
from evaluate.bleu import bleu
from evaluate.distinct_n import distinct_n_corpus_level
import pandas as pd


def resume_from_checkpoint(checkpoint_dir, model):
    # full_checkpoint
    checkpoint_name = os.path.join(checkpoint_dir, "pytorch_model.bin")
    # lora_checkpoint
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(checkpoint_dir, "adapter_model.bin")
    # load model
    if os.path.exists(checkpoint_name):
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")


def evaluate(
        instruction,
        model,
        prompter,
        tokenizer,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        device="cuda",
        **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompter.get_response(output)


# 包含知识扩充的evaluate
def evaluate_knowledge(
        knowledge,
        instruction,
        model,
        prompter,
        tokenizer,
        temperature=0.5,
        top_p=0.8,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        device="cuda",
        **kwargs,
):
    full_knowledge, full_qa = prompter.generate_prompt(knowledge, instruction)
    # 若要生成更优质的回答，不能进行截断处理
    inputs = tokenizer(full_knowledge, full_qa, padding=False, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=1.2,
        force_words_ids=[[2]],
        **kwargs,
    )
    with torch.no_grad():
        # global current_token_len
        # current_token_len = len(input_ids)
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            # logits_processor=logits_processor,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompter.get_response(output)


class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_folder)
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def longestDupSubstring(s: str) -> str:
    left = 0
    right = 1
    res = ""
    n = len(s)
    while right < n:
        if s[left:right] in s[left + 1:]:
            if right - left > len(res):
                res = s[left:right]
            right += 1
            continue
        left += 1
        if left == right:
            right += 1
    return res


def remove_repeat(raw_response, threshold=10, repeat_nums=3):
    """
    :param raw_response:
    :param threshold: 可以容忍的最短公共子序列长度
    :param repeat_nums: 去重次数
    :param
    :return:
    """
    sentence = raw_response
    # 如果原本的句子有结束符号，说明当前句子结构完整，那么不需要处理。
    if raw_response[-1] == '>':
        return sentence
    while repeat_nums > 0 and len(sentence) > 0:
        repeat_nums -= 1
        repeat = longestDupSubstring(sentence)
        if len(repeat) > threshold:
            # 保留第一个，删除第二个及以后的
            first_index = sentence.find(repeat)
            second_index = sentence.find(repeat, first_index + len(repeat))
            # 说明最后一直在重复，那么删除first_index之后所有的字符串
            if second_index == -1:
                sentence = sentence[:first_index]
            else:
                sentence = sentence[:second_index]
        else:
            break
    # 最后被删除到很少的数据了，说明一直存在重复信息。保留原始数据
    if len(sentence) < threshold:
        sentence = raw_response
    res = re.findall(r'[,.;!?，。！？；]', sentence)
    # 当前句子不是以标点符号结尾，并且前面存在很多标点符号，那么就删除到最后一个标点符号位置
    if len(res) > 1 and res[-1] != sentence[-1]:
        right_sign_index = sentence.rfind(res[-1])
        sentence = sentence[:right_sign_index + 1]
    # 最后一位的处理，添加句号，添加</s>
    if sentence[-1] != '>':
        if sentence[-1] in [',', '，']:
            sentence = sentence[:-1] + '。</s>'
        elif sentence[-1] in ['.', '。', '!', '！']:
            sentence += '</s>'
        else:
            sentence += '。</s>'
    return sentence


# 调用chatGPT接口
def use_ChatGPT(dialogue):
    client = OpenAI(api_key="")
    template = \
        "我希望你扮演一名专业的眼科医生，来进行多轮次对话的整理工作。我将会向你提供一个医生和病人之间的多轮次会话，请你重新生成一个相同主题的多轮次对话：\n" \
        "原始对话：\n" \
        f"{dialogue} \n" \
        "注意事项：\n" \
        "1.病人在阐述病情时，需要回答医生提出的问题；\n" \
        "2.医生在进行回复时，应注意规范医疗专业用词，并向病人进行病情咨询；\n" \
        "3.每一轮次的回复长度在30个字符以内，对话内容不要口语化；\n" \
        "4.病人和医生进行10到15轮次以上的对话；\n" \
        "5.医生和病人的回复应与当前诊断的眼科疾病强相关；\n" \
        "6.病人和医生交替进行对话；\n" \
        "7.病人首次对话时，应该向医生简单描述疾病症状，不宜过长；\n" \
        "8.以原始对话的json格式返回生成的对话结果。"
    messages = [
        {"role": "user",
         "content": template
         }
    ]
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    generated_content = completion.choices[0].message
    print(generated_content)


# 用户根据用户疾病生成疾病治疗方案
def load_knowledge_data(path, max_length=400):
    with open(path, "r", encoding="utf-8") as disease_file:
        disease_data = disease_file.readlines()
        knowledge_context = []
        for single_data in disease_data:
            data_dict = json.loads(single_data)
            data = str()
            if len(data_dict["summary"]) > 0:
                data += f"概述:{data_dict['summary']}"
            if len(data_dict["treatment"]) > 0:
                data += f"治疗:{data_dict['treatment']}"
            if len(data_dict["medicine"]) > 0:
                data += f"推荐药品:{data_dict['medicine']}"
            if len(data_dict["symptom"]) > 0:
                data += f"症状:{data_dict['symptom']}"
            knowledge_context.append(data)
    return knowledge_context


# 用于根据当前用户叙述判断最相近的疾病
def load_disease_data(path):
    with open(path, "r", encoding="utf-8") as disease_file:
        disease_data = disease_file.readlines()
        disease_data_json = [(f"{json.loads(single_data)['disease']} "
                              f"{json.loads(single_data)['symptom']} {json.loads(single_data)['summary']}")
                             for single_data in disease_data]
    return disease_data_json


# 数据输入类型统一为[[1,2,3,4,5],[6,7,8,9,10]]
# 验证函数
def compute_metrics(labels, preds, verbose=True):
    rouge_scores = get_rouge_l(preds, labels)
    bleu_1 = bleu(labels, preds, 1)
    bleu_2 = bleu(labels, preds, 2)
    bleu_3 = bleu(labels, preds, 3)
    bleu_4 = bleu(labels, preds, 4)
    distinct_1 = distinct_n_corpus_level(preds, 1)
    distinct_2 = distinct_n_corpus_level(preds, 2)
    distinct_3 = distinct_n_corpus_level(preds, 3)
    distinct_4 = distinct_n_corpus_level(preds, 4)
    result_dict = {
        'rouge-l': ['%.4f' % rouge_scores],
        'bleu_1': ['%.4f' % bleu_1],
        'bleu_2': ['%.4f' % bleu_2],
        'bleu_3': ['%.4f' % bleu_3],
        'bleu_4': ['%.4f' % bleu_4],
        'distinct-1': ['%.4f' % distinct_1],
        'distinct-2': ['%.4f' % distinct_2],
        'distinct-3': ['%.4f' % distinct_3],
        'distinct-4': ['%.4f' % distinct_4],
    }
    # 打印成表格样式
    if verbose:
        pd.set_option('display.max_columns', None)
        aa = pd.DataFrame(result_dict)
        print(aa)
    return result_dict
