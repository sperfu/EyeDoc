import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_chat import message
import openai
import os
import sys


import openai
from openai import OpenAI
import pdb
import json

from A_generate_qa_test import Generate

# 用户名和密码的默认值
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_BASE_URL"] = "your_url" ## could be comment

client = OpenAI()
VERSION = 0.1

tinyllama_knowledge_param = {
        "base_model": "tinyLlama",
        "test_count": "500",  # 测试集的数量，如果测试集不存在，会生成该测试集。
        "lora_model": "save_model/checkpoint-3648",
        "knowledge_path": "data/eye_disease_knowledge_extract.json",
        "disease_model": "text2vec-similar",
        "output_text": "gen_100.txt",
        "template": {
            "prompt_input": "下面是一个眼部疾病相关的问题，请运用医学知识来正确回答提问。这里提供了一些可以参考的消息。"
                            "\n### 参考信息:\n{knowledge}"
                            "\n### 问题:\n{instruction}"
                            "\n### 回答:\n",
            "response_split": "### 回答:"
        }
    }

# 检查会话状态中是否有登录状态，如果没有，初始化为 False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'past' not in st.session_state:
    st.session_state.past = {
        
        "EyeDoc-chat 眼科医生": []
    }

if 'generated' not in st.session_state:
    st.session_state.generated = {
        
        "EyeDoc-chat 眼科医生": []
    }

def on_input_change():
    user_input = st.session_state.user_input
    current_page = st.session_state.selected_page
    
    st.session_state.past[current_page].append(user_input)
    
    role_description = "你是一个专业的农业育种专家，我现在需要给你一段摘要，请你帮我根据这段摘要内容，提取出来1-2条问题并生成对应的回答，要求：1. 回答必须专业，学术化，准确无误.尽量全面详细,回答充分一些，回答按照问题1: \n，回答1:\n，问题2: \n，回答2:\n进行,要换行 2. 问题中要以科普的口吻为主，不能出现本研究，研究了，研究结果表明等等字眼。3. 如果摘要内容比较短，则根据你的掌握知识，扩展一下具体内容并形成专业化问答;请以markdown格式输出"
    # 创建消息列表，包括系统消息和用户消息
    messages = [
        {"role": "system", "content": role_description},
        {"role": "user", "content": user_input}
    ]
    
    # 调用OpenAI的ChatCompletion API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", #gpt-4o-2024-05-13
        messages=messages,
        n=1,
        stop=None,
        temperature=0.7
    )

    bot_response = response.choices[0].message.content
    #st.session_state.generated.append({'type': 'normal', 'data': bot_response})
    st.session_state.generated[current_page].append({'type': 'markdown', 'data': bot_response})
    #st.session_state.generated.append("The messages from Bot\nWith new line")
    st.session_state.user_input = ''

def on_input_change_eyedoc():
    user_input = st.session_state.user_input
    current_page = st.session_state.selected_page
    
    st.session_state.past[current_page].append(user_input)
    
    role_description = "你是一个专业的眼科医生，请帮我根据患者的提问，给出医学专业的诊疗意见并开具处方，要求：1. 回答必须专业，学术化，准确无误.尽量全面详细,回答充分一些，自然一些;请以markdown格式输出"
    # 创建消息列表，包括系统消息和用户消息
    messages = [
        {"role": "system", "content": role_description},
        {"role": "user", "content": user_input}
    ]
    
    # 调用OpenAI的ChatCompletion API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", #gpt-4o-2024-05-13
        messages=messages,
        n=1,
        stop=None,
        temperature=0.7
    )

    bot_response = response.choices[0].message.content
    #st.session_state.generated.append({'type': 'normal', 'data': bot_response})
    st.session_state.generated[current_page].append({'type': 'markdown', 'data': bot_response})
    #st.session_state.generated.append("The messages from Bot\nWith new line")
    st.session_state.user_input = ''

def on_input_change_eyedoc_tinnyllama(generate):
    
    user_input = st.session_state.user_input
    current_page = st.session_state.selected_page
    
    st.session_state.past[current_page].append(user_input)
    
    # role_description = "你是一个专业的眼科医生，请帮我根据患者的提问，给出医学专业的诊疗意见并开具处方，要求：1. 回答必须专业，学术化，准确无误.尽量全面详细,回答充分一些，自然一些;请以markdown格式输出"
    # # 创建消息列表，包括系统消息和用户消息
    # messages = [
    #     {"role": "system", "content": role_description},
    #     {"role": "user", "content": user_input}
    # ]
    
    
    # question = "现在发现拍照眼睛一个大一个小，左眼无神，看上去空洞的，去看过眼科专家，检查不出来什么问题，滴了治干眼症的眼药水得到好转了，左眼也正常了，但是拍照看着总是很奇怪"
    # 调用OpenAI的ChatCompletion API
    raw_response, bot_response = generate.generate_qa(question=user_input, remove_repeat=True)
    

    
    st.session_state.generated[current_page].append({'type': 'markdown', 'data': bot_response})
    
    st.session_state.user_input = ''

def on_btn_click():
    current_page = st.session_state.selected_page
    st.session_state.past[current_page] = []
    st.session_state.generated[current_page] = []

def login_page():
    with st.form("login_form"):
        st.title("登录")
        username = st.text_input("用户名", value="")
        password = st.text_input("密码", value="", type="password")
        submit = st.form_submit_button("登录")

        if submit:
            if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
                st.success("登录成功！")
                # 更新会话状态为已登录
                st.session_state.logged_in = True
                st.rerun()  # 重新运行脚本以显示主页面
            else:
                st.error("用户名或密码错误，请重新输入。")



def chat_page_eyedoc(generate):
    st.title("EyeDoc-chat placeholder")
    current_page = "EyeDoc-chat 眼科医生"
    chat_placeholder = st.empty()

    with chat_placeholder.container():    
        for i in range(len(st.session_state['generated'][current_page])):                
            message(st.session_state['past'][current_page][i], is_user=True, key=f"{i}_user_{current_page}")
            message(
                st.session_state['generated'][current_page][i]['data'], 
                key=f"{i}_{current_page}", 
                allow_html=True,
                is_table=True if st.session_state['generated'][current_page][i]['type']=='table' else False
            )
    
        st.button("Clear message", on_click=on_btn_click)

    with st.container():
        # st.text_input("User Input:", on_change=on_input_change_eyedoc, key="user_input")
        st.text_input("User Input:", on_change=lambda: on_input_change_eyedoc_tinnyllama(generate), key="user_input")


def main_page(generate):
    is_lite = "lite" in sys.argv
    st.set_page_config(
        "EyeDoc-chat WebUI",
        os.path.join("st_chat_img", "images.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'About': f"""欢迎使用 EyeDoc-chat WebUI {VERSION}！"""
        }
    )
    #pdb.set_trace()
    pages = {
        
        "EyeDoc-chat 眼科医生": {
            "icon": "hdd-stack",
            "func": lambda: chat_page_eyedoc(generate),
        },
    }

    with st.sidebar:
        st.image(
            os.path.join(
                "st_chat_img",
                "images.png"
            ),
            use_column_width=True
        )
        st.caption(
            f"""<p align="right">当前版本：{VERSION}</p>""",
            unsafe_allow_html=True,
        )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            default_index=default_index,
        )

        st.session_state.selected_page = selected_page

    if selected_page in pages:
        pages[selected_page]["func"]()

if __name__ == "__main__":
    generate = Generate(tinyllama_knowledge_param)
    if st.session_state.logged_in:
        if 'selected_page' not in st.session_state:
            st.session_state.selected_page = "EyeDoc-chat 眼科医生"  # 默认页面
        main_page(generate)
    else:
        login_page()