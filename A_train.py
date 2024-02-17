# @Author       : Duhongkai
# @Time         : 2024/1/3 18:16
# @Description  : 训练模型

import os
import sys
import torch
import transformers
import datasets
from peft import (
    PrefixTuningConfig,
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from utils import util
from utils.prompter import Prompter
from utils import mail

from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 防止死锁
# 一些解释 load_in_8bit, prepare_model_for_int8_training, get_peft_model
# https://zhuanlan.zhihu.com/p/651338142
#

class Train:
    def __init__(self, a_param):
        self.param = a_param
        self.print_param()
        self.prompter = Prompter(self.param['template'])
        self.load_wandb()
        self.tokenizer, self.model = self.load_model()
        self.train_dataset, self.val_dataset = self.init_dataset(count=self.param["val_set_size"])
        self.train()

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(f"{self.param['base_model']}")
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            self.param['base_model'],
            load_in_8bit=True,
            torch_dtype=torch.float16,
            # device_map={"": torch.cuda.current_device()},
            device_map="auto",
        )
        model = prepare_model_for_int8_training(model)
        lora_config = LoraConfig(
            r=self.param['lora_r'],
            lora_alpha=self.param['lora_alpha'],
            target_modules=self.param['lora_target_modules'],
            lora_dropout=self.param['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        prefix_tuning_config = PrefixTuningConfig(
            peft_type="PREFIX_TUNING",
            task_type="CAUSAL_LM",
            num_virtual_tokens=20,
            token_dim=768,
            num_transformer_submodules=1,
            num_attention_heads=12,
            num_layers=12,
            encoder_hidden_size=768,
        )
        # prefix_tuning_model = get_peft_model(model, prefix_tuning_config)
        lora_model = get_peft_model(model, lora_config)
        lora_model.print_trainable_parameters()
        print("**************")
        print(lora_model.hf_device_map)
        # 断点续传，不需要，训练器已经集成
        # if self.param["resume_from_checkpoint"]:
        #     util.set_peft_model_state_dict(self.param["resume_from_checkpoint"], lora_model)
        return tokenizer, lora_model

    def init_dataset(self, ratio=0.1, count=0):
        data = datasets.load_dataset('json', data_files=self.param["data_path"])
        knowledge_data = util.load_knowledge_data(path=self.param["knowledge_path"], max_length=self.param["max_len"])
        val_count = int(len(data["train"]) * ratio) if count == 0 else count
        train_val = data["train"].train_test_split(test_size=val_count, shuffle=True, seed=42)

        def generate_and_tokenize_prompt(data_point):
            """
            数据标准化处理，类似于dataset
            """
            full_knowledge, full_qa = self.prompter.generate_prompt(
                knowledge_data[data_point["disease_index"]],    # 知识源
                data_point["instruction"],
                data_point["output"],
            )
            tokenized_full_prompt = self.tokenize(full_knowledge, full_qa)
            if not self.param["train_on_inputs"]:  # if False, masks out inputs in loss, input不会添加到训练指标中
                # 数据截断
                tokenized_output_prompt = self.tokenizer(data_point["output"], truncation=True, max_length=self.param['max_len'], padding=False, return_tensors=None)
                user_prompt_len = len(tokenized_full_prompt["input_ids"]) - len(tokenized_output_prompt["input_ids"])
                # 将用户的输入部分进行mask，避免计算loss
                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            return tokenized_full_prompt
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

    # 编译之后，添加eos，对应的修改input_ids、attention_mask和labels
    def tokenize(self, knowledge, qa, add_eos_token=True):
        # 在进行截断时，只截断第一个(第一个是知识源)
        result = self.tokenizer(knowledge, qa, truncation="only_first", max_length=self.param['max_len'],
                                padding=False, return_tensors=None)
        # 如果不行，截断第二个
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
                eval_steps=32,          # 每32步进行一次验证
                save_steps=32,          # 每32步进行一次保存
                output_dir=self.param["output_dir"],
                save_total_limit=10,
                load_best_model_at_end=True,
                ddp_find_unused_parameters=False,
                group_by_length=False,
                report_to="wandb",
                run_name=self.param["standard_name"],
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            callbacks=[util.SavePeftModelCallback],
        )
        self.model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        # with torch.autocast("cuda"):
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
        "standard_name": "lora_bloom_knowledge_eyeQA",
        # model/data params
        # "base_model": "Llama-2-7b-chinese-chat",  # the only required argument
        "base_model": "bloom-zh-3b",  # the only required argument
        "data_path": "./data/eye_QA_knowledge.json",
        "knowledge_path": "./data/eye_disease_knowledge_processed.json",
        "output_dir": "./bloom_output/",

        # training hyperparams
        "batch_size": 32,  # 每一个batch_size会进行参数更新
        "micro_batch_size": 32,  # 最小的一个batch_size，不进行参数更新
        "num_epochs": 10,
        "learning_rate": 3e-4,
        # "max_len": 256,  # max_length    # 经过测验，99%的数据长度在250范围之内
        "max_len": 512,  # max_length      # 添加知识源后，长度扩充
        "val_set_size": 500,  # 验证集数量

        # lora hyperparams
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["query_key_value"],         # bloom-3b
        # "lora_target_modules": ["q_proj", "v_proj"],      # llama

        # llm hyperparams
        "train_on_inputs": False,  # if False, masks out inpu ts in loss
        "group_by_length": False,  # faster, but produces an odd training loss curve

        # wandb params
        "wandb_project": "llama_med",
        "wandb_watch": "",  # options: false | gradients | all
        "wandb_log_model": "",  # options: false | true
        "resume_from_checkpoint": None,  # either training checkpoint or final adapter

        # prompt
        "template": {
        "prompt_input": "下面是一个眼部疾病相关的问题，请运用医学知识来正确回答提问。这里提供了一些可以参考的消息。"
                        "\n### 参考信息:\n{knowledge}"
                        "\n### 问题:\n{instruction}"
                        "\n### 回答:\n",
        "response_split": "### 回答:"
        }
    }
    Train(param)
