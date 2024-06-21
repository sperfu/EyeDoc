
<div align="center">
<h1>
  EyeDoc
</h1>
<h2>ophthalmic consultation foundation model</h2>
</div>
<p align="center">
📝 <a href="https://arxiv.org/" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/" target="_blank">Hugging Face</a> • 🧩 <a href="https://github.com/sperfu/EyeDoc" target="_blank">Github</a>
</p>



## ✨ Recent News

- [2/18/2024] The first version of the model was released.

## ⚡ Introduction

EyeDoc is the first open-source large language model focused on ophthalmic diseases. Our goal in developing EyeDoc is to create a more specialized large language model for the specific medical consultation scenario of ophthalmic diseases. Overall, compared to other medical large language models, our contributions are as follows:

1. We have collected over 40,000 single-turn QA dialogues and nearly 9,000 multi-turn dialogues related to ophthalmic diseases. To standardize the multi-turn dialogue data, we used **gpt-3.5-turbo** for data cleaning.

2. We have gathered knowledge information on 519 common ophthalmic diseases and constructed a specialized knowledge base for auxiliary diagnosis of ophthalmic diseases.

3. We have fully considered the knowledge differences and language characteristics of doctors and patients during consultations, and based on this, we separately represented the features for doctors and patients.

![Model Pipeline](assets/img/eye_main.jpg)


## 🤖 Installation

```
python==3.9.0
torch==2.1.2
transformers==4.35.2
peft==0.7.1
accelerate==0.25.0
bitsandbytes==0.42.0
rouge_chinese
nltk
```

## 💭 Prepare

EyeDoc基于大语言模型微调而来，在训练之前请先配置或下载大语言模型基座。

| 参数规模 | 大语言模型名称                                               |
| -------- | ------------------------------------------------------------ |
| 1B       | [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| 3B       | [bloom-zh-3b-chat](https://huggingface.co/ikala/bloom-zh-3b-chat) |
| 7B       | [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |

## ⚒️ Train

###  1. QA Train

```python
python A_train.py
# 更多模型超参数调整见main函数内
```


###  2.  Multi-turn Training

```python
python A_train_doc_specific.py
# 更多模型超参数调整见main函数内
```

##  🧐 Evaluation

```python
python A_evaluate.py
# 更多模型超参数调整见main函数内
```

## 🚀 Generate

```
python A_generate.py	# 模型响应生成
python A_Flask_web.py	# 模型api接口调用
```

## 🌐 Deployment

To deploy the EyeDoc model using Streamlit, follow these steps:

### 1. Install the required environment

First, ensure you have Python installed (preferably Python 3.9). Then, install Streamlit and other necessary packages:

```bash
pip install streamlit
pip install -r requirements.txt  # Ensure all dependencies listed in the requirements file are installed
```

### 2. Start the Streamlit service

Run the Streamlit application using the following command:

```bash
streamlit run st_chat_login.py
```

This will start the Streamlit service, and you can access the web interface through the provided local URL (e.g., http://localhost:8501).