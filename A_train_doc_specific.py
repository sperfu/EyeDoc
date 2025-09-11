# @Author       : Duhongkai
# @Time         : 2024/1/26 14:35
# @Description  :
# 突出医患角色信息的医疗问答系统, Lora+prefix-tuning

import os
import sys

MODULE_PATH = os.path.abspath("..")
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
import torch
import transformers
import datasets
from peft import (
    PrefixTuningConfig,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training, prepare_model_for_int8_training, inject_adapter_in_model, PeftModel
)
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, BertModel, BertTokenizer
from dataset import data_collator
from utils import util
from utils.prompter import MultiPrompter
from merge_peft_model import MyPeftModelForCausalLM
from utils import mail

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 防止死锁


# 一些解释 load_in_8bit, prepare_model_for_int8_training, get_peft_model
# https://zhuanlan.zhihu.com/p/651338142
#

class Train:
    def __init__(self, a_param):
        self.param = a_param
        self.print_param()
        self.prompter = MultiPrompter(self.param['template'])
        self.load_wandb()
        self.tokenizer, self.model, self.roberta_tokenizer, self.roberta_model = self.load_model()
        self.train_dataset, self.val_dataset = self.init_dataset(count=self.param["val_set_size"])
        self.train()

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(f"{self.param['base_model']}")
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"  # 裁剪左侧的对话文本
        # 为了兼容框架，向peftmodel中添加我们实现的peft方式(lora+prefix—tuning)
        peft.MODEL_TYPE_TO_PEFT_MODEL_MAPPING['CAUSAL_LM_MERGE'] = MyPeftModelForCausalLM
        # 使用4bit量化模型
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16,
        # )
        model = AutoModelForCausalLM.from_pretrained(
            self.param['base_model'],
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        # 加载训练好的模型
        # eye_model = PeftModel.from_pretrained(
        #     model,
        #     f'{self.param["output_dir"]}prefix_model',
        #     torch_dtype=torch.float16,
        #     adapter_name="prefix_model",
        # )
        # lora_model = PeftModel.from_pretrained(
        #     eye_model,
        #     f'{self.param["output_dir"]}lora_model',
        #     torch_dtype=torch.float16,
        #     adapter_name="lora_model",
        # )

        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            task_type="CAUSAL_LM_MERGE",
            r=self.param['lora_r'],
            lora_alpha=self.param['lora_alpha'],
            target_modules=self.param['lora_target_modules'],
            lora_dropout=self.param['lora_dropout'],
            bias="none",
        )
        prefix_tuning_config = PrefixTuningConfig(
            task_type="CAUSAL_LM_MERGE",
            num_virtual_tokens=1,
            inference_mode=False,
            # token_dim=model.config.hidden_size,  # hidden_size - llama
            token_dim=256,  # hidden_size - llama
            num_attention_heads=model.config.num_key_value_heads,  # 这里使用的是key_value的head数目, 4 - llama
        )
        # peft_model = get_peft_model(model, lora_config)
        peft_model = get_peft_model(model, prefix_tuning_config, adapter_name="prefix_model")
        peft_model = inject_adapter_in_model(lora_config, peft_model, adapter_name="lora_model")
        # 添加prefix_tuning的梯度，注意bert需要freeze(现在prefix tuning中不包含bert)
        for n, p in peft_model.named_parameters():
            if 'prompt_encoder' in n:
                p.requires_grad = True
        peft_model.print_trainable_parameters()

        # 参数量较小，不需要量化
        roberta_tokenizer = BertTokenizer.from_pretrained(self.param["roberta_model"])
        # 添加gpt模型中的end_eos token和start_eos
        roberta_tokenizer.add_special_tokens({'additional_special_tokens': ["</e>"]})
        roberta_tokenizer.truncation_side = "left"  # 过长的话，裁剪左侧文本
        roberta_model = BertModel.from_pretrained(self.param["roberta_model"])
        return tokenizer, peft_model, roberta_tokenizer, roberta_model

    def init_dataset(self, ratio=0.1, count=0):
        data = datasets.load_dataset('json', data_files=self.param["data_path"])
        val_count = int(len(data["train"]) * ratio) if count == 0 else count
        train_val = data["train"].train_test_split(test_size=val_count, shuffle=True, seed=42)

        def generate_and_tokenize_prompt(data_point):
            # gpt多轮次
            full_prompt = self.prompter.generate_prompt(
                history=data_point["history"],
                output=data_point["output"],
            )
            tokenized_full_prompt = self.tokenize(full_prompt)
            if not self.param["train_on_inputs"]:  # if False, masks out inputs in loss, input不会添加到训练指标中
                tokenized_output_prompt = self.tokenizer(data_point["output"], truncation=True,
                                                         max_length=self.param['max_len'], padding=False,
                                                         return_tensors=None)
                # 需要注意，有的模型会在句子前面添加<s>，那么此时user_prompt_len需要+2处理(tinyllama发现该问题)
                user_prompt_len = len(tokenized_full_prompt["input_ids"]) - len(tokenized_output_prompt["input_ids"])
                # 将用户的输入部分进行mask，避免计算loss
                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                             user_prompt_len:]
            # roberta
            doctor_tokenizer = self.roberta_tokenizer(data_point["doctor"], truncation=True,
                                                      max_length=self.param["max_len"] // 2, padding=False,
                                                      return_tensors=None)
            patient_tokenizer = self.roberta_tokenizer(data_point["patient"], truncation=True,
                                                       max_length=self.param["max_len"] // 2, padding=False,
                                                       return_tensors=None)

            # return {"input_ids": tokenized_full_prompt["input_ids"], "attention_mask": tokenized_full_prompt["attention_mask"],
            #         "labels": tokenized_full_prompt["labels"],
            #         "doctor": doctor_tokenizer, "patient": patient_tokenizer}
            return {"full_prompt": tokenized_full_prompt, "doctor": doctor_tokenizer, "patient": patient_tokenizer}
            # return tokenized_full_prompt

        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
        return train_data, val_data

    def print_param(self):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"Training Alpaca-LoRA model with params:\n"
                f"base_model: {self.param['base_model']}\n"
                f"data_path: {self.param['data_path']}\n"
                f"output_dir: {self.param['output_dir']}\n"
                f"batch_size: {self.param['batch_size']}\n"
                f"micro_batch_size: {self.param['micro_batch_size']}\n"
                f"num_epochs: {self.param['num_epochs']}\n"
                f"learning_rate: {self.param['learning_rate']}\n"
                f"max_len: {self.param['max_len']}\n"
                f"val_set_size: {self.param['val_set_size']}\n"
                f"lora_r: {self.param['lora_r']}\n"
                f"lora_alpha: {self.param['lora_alpha']}\n"
                f"lora_dropout: {self.param['lora_dropout']}\n"
                f"lora_target_modules: {self.param['lora_target_modules']}\n"
                f"train_on_inputs: {self.param['train_on_inputs']}\n"
                f"group_by_length: {self.param['group_by_length']}\n"
                f"wandb_project: {self.param['wandb_project']}\n"
                f"wandb_watch: {self.param['wandb_watch']}\n"
                f"wandb_log_model: {self.param['wandb_log_model']}\n"
                f"resume_from_checkpoint: {self.param['resume_from_checkpoint'] or False}\n"
                f"template: {self.param['template']}\n"
            )

    def load_wandb(self):
        os.environ["WANDB_PROJECT"] = self.param["wandb_project"]
        os.environ["WANDB_MODE"] = "offline"

    # 编译之后，添加eos，对应的修改input_ids、attention_mask和labels
    def tokenize(self, full_prompt, add_eos_token=True):
        result = self.tokenizer(full_prompt, truncation=True, max_length=self.param['max_len'], padding=False,
                                return_tensors=None)
        if (result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.param['max_len']
                and add_eos_token):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def train(self):
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.param["micro_batch_size"],
                gradient_accumulation_steps=self.param['batch_size'] // self.param['micro_batch_size'],
                warmup_ratio=0.1,
                num_train_epochs=self.param["num_epochs"],
                learning_rate=self.param["learning_rate"],
                fp16=True,
                logging_steps=8,
                optim="adamw_torch",
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=2,  # 每32步进行一次验证
                save_steps=2,  # 每32步进行一次保存
                output_dir=self.param["output_dir"],
                save_total_limit=10,
                load_best_model_at_end=True,
                ddp_find_unused_parameters=False,
                group_by_length=False,
                report_to="wandb",
                run_name=self.param["standard_name"],
                label_names=["labels"],
            ),
            data_collator=data_collator.DataCollatorForSeq2Seq(
                self.tokenizer, self.roberta_tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            # data_collator=transformers.DataCollatorForSeq2Seq(
            #     self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            # ),
            callbacks=[util.SavePeftModelCallback],
        )

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     self.model = torch.compile(self.model)

        with torch.autocast("cuda"):  # 解决量化报错
            trainer.train(resume_from_checkpoint=self.param["resume_from_checkpoint"])
        # 保存最终模型
        self.model.save_pretrained(self.param["output_dir"])
        print("train over!")
        # 邮件通知
        subject = '模型训练完成'
        body = 'llama大模型微调完成'
        mail.send_msg(subject, body)


if __name__ == "__main__":
    param = {
        "standard_name": "tinyllama_specific",

        # model/data params
        # "base_model": "Llama-2-7b-chinese-chat",  # the only required argument
        # "base_model": "bloom-zh-3b",  # the only required argument
        "base_model": "tinyLlama",  # the only required argument
        "roberta_model": "diagBERT",  # 医患关系表征使用的模型
        "data_path": "./data/dingxiang_multi_dialog_processed_mini.json",
        "output_dir": "./tinyLlama_output/",

        # training hyperparams
        "batch_size": 32,  # 每一个batch_size会进行参数更新
        "micro_batch_size": 1,  # 最小的一个batch_size，不进行参数更新
        "num_epochs": 10,
        "learning_rate": 3e-4,
        "max_len": 1024,  # max_length。经验证，95%在1063，后续有所有的数据后再修改
        "val_set_size": 100,  # 验证集数量

        # lora hyperparams
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        # "lora_target_modules": ["query_key_value"],  # bloom-3b
        "lora_target_modules": ["q_proj", "v_proj"],  # llama、tinyllama

        # llm hyperparams
        "train_on_inputs": False,  # if False, masks out inpu ts in loss
        "group_by_length": False,  # faster, but produces an odd training loss curve

        # wandb params
        "wandb_project": "llama_med",
        "wandb_watch": "",  # options: false | gradients | all
        "wandb_log_model": "",  # options: false | true
        "resume_from_checkpoint": None,  # either training checkpoint or final adapter

        # prompt
        # 设计多轮次的提示信息时，加入你好，我是你的眼科医生，请问你有什么不适症状的话语
        "template": {
            "prompt_input": "<医生>：您好，我是您的眼科医生。请问您有哪些眼睛不适症状？</s>{history}<医生>：",
            "response_split": "<医生>："
        }
    }
    Train(param)
