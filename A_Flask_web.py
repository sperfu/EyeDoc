# @Author       : Duhongkai
# @Time         : 2024/1/16 11:32
# @Description  : Flask界面，方便使用
from A_generate import Generate
from flask import Flask, request, jsonify

from utils.ResponseData import ResponseData

app = Flask(__name__)


@app.route('/llama_qa', methods=['GET'])
def llama():
    question = request.args.get("question")
    if not question:
        response = ResponseData(status=400, message="error", data="问题为空")
    else:
        llama_7b_param = {
            "standard_name": "lora_bloom_eyeQA",
            "base_model": "Llama-2-7b-chinese-chat",  # the only required argument
            "lora_model": "lora/llama-eye-7b",
            "prompt_template_name": "med_template",
        }
        generate = Generate(llama_7b_param)
        answer = generate.generate_qa(question)
        response = ResponseData(data={"question": question, "answer": answer})
    return jsonify(response.to_dict())


@app.route('/bloom_qa', methods=['GET'])
def bloom():
    question = request.args.get("question")
    if not question:
        response = ResponseData(status=400, message="error", data="问题为空")
    else:
        bloom_3b_param = {
            "standard_name": "lora_bloom_eyeQA",
            "base_model": "bloom-zh-3b",
            "lora_model": "lora/bloom-eye-3b",
            "prompt_template_name": "med_template",
        }
        generate = Generate(bloom_3b_param)
        answer = generate.generate_qa(question)
        response = ResponseData(data={"question": question, "answer": answer})
    return jsonify(response.to_dict())


@app.route('/bloom_knowledge_qa', methods=['GET'])
def bloom_knowledge():
    question = request.args.get("question")
    if not question:
        response = ResponseData(status=400, message="error", data="问题为空")
    else:
        bloom_3b_param = {
            "standard_name": "lora_bloom_eyeQA",
            "base_model": "bloom-zh-3b",
            "lora_model": "lora/bloom-eye-3b-knowledge/checkpoint-5408",
            "template": {
                "prompt_input": "下面是一个眼部疾病相关的问题，请运用医学知识来正确回答提问。这里提供了一些可以参考的消息。"
                                "\n### 参考信息:\n{knowledge}"
                                "\n### 问题:\n{instruction}"
                                "\n### 回答:\n",
                "response_split": "### 回答:"
            }
        }
        generate = Generate(bloom_3b_param)
        answer = generate.generate_qa(question)
        response = ResponseData(data={"question": question, "answer": answer})
    return jsonify(response.to_dict())


if __name__ == '__main__':
    app.run(debug=True, port=4590)
