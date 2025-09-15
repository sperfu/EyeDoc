import streamlit as st
from streamlit_option_menu import option_menu
import openai
from openai import OpenAI
import os
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import sys
import json

# Correctly import your Generate class from the local file
from A_generate import Generate

# --- CONFIGURATION AND INITIALIZATION ---

# Define avatar paths
USER_AVATAR_PATH = "st_chat_img/user.png"  # Using relative paths is often more robust
ASSISTANT_AVATAR_PATH = "st_chat_img/images.png"

# User credentials
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"

# OpenAI API key (can be a placeholder if not used)
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_BASE_URL"] = "your_url"

client = OpenAI()
VERSION = 0.1

# Local model parameters from your script
tinyllama_knowledge_param = {
    "base_model": "tinyLlama",
    "test_count": "500",
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

# --- UI HELPER FUNCTIONS ---

def image_to_base64(image_pil):
    """Converts a PIL image to a base64 string."""
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def image_file_to_base64(path):
    """Reads an image file and converts it to a resized base64 string."""
    try:
        image = Image.open(path).resize((40, 40))
        return image_to_base64(image)
    except (FileNotFoundError, Exception):
        return generate_default_avatar("#CCCCCC")

def generate_default_avatar(color):
    """Generates a simple default avatar image."""
    img = Image.new("RGB", (40, 40), color)
    draw = ImageDraw.Draw(img)
    draw.ellipse((5, 5, 35, 35), fill=(255, 255, 255))
    return image_to_base64(img)

def add_chat_styles():
    """Adds custom CSS for chat bubbles and avatars."""
    st.markdown("""
    <style>
    .chat-container { display: flex; align-items: flex-start; margin: 15px 0; gap: 10px; }
    .user-chat { flex-direction: row-reverse; }
    .model-chat { flex-direction: row; }
    .chat-message-box { max-width: 70%; padding: 12px 16px; border-radius: 18px; word-wrap: break-word; line-height: 1.4; }
    .user-chat .chat-message-box { background-color: #007bff; color: white; border-bottom-right-radius: 4px; }
    .model-chat .chat-message-box { background-color: #f1f3f5; color: #333; border-bottom-left-radius: 4px; border: 1px solid #e9ecef; }
    .chat-avatar { width: 45px; height: 45px; border-radius: 50%; object-fit: cover; flex-shrink: 0; border: 2px solid #e9ecef; }
    .user-chat .chat-avatar { border-color: #007bff; }
    .model-chat .chat-avatar { border-color: #28a745; }
    </style>
    """, unsafe_allow_html=True)

def display_chat_message(role, content):
    """Displays a single chat message using the custom HTML format."""
    escaped_content = content.replace('<', '&lt;').replace('>', '&gt;')
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-container user-chat">
            <div class="chat-message-box">{escaped_content}</div>
            <img class="chat-avatar" src="data:image/png;base64,{user_avatar}">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-container model-chat">
            <img class="chat-avatar" src="data:image/png;base64,{model_avatar}">
            <div class="chat-message-box">{escaped_content}</div>
        </div>
        """, unsafe_allow_html=True)

# --- SESSION STATE & LOGIC ---

# Load avatars
user_avatar = image_file_to_base64(USER_AVATAR_PATH)
model_avatar = image_file_to_base64(ASSISTANT_AVATAR_PATH)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'messages' not in st.session_state:
    st.session_state.messages = []


def on_input_change(generate):
    """Callback to handle user input and generate a response."""
    user_input = st.session_state.user_input
    if not user_input:
        return

    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate bot response using your local model
    try:
        raw_response, bot_response = generate.generate_qa(question=user_input, remove_repeat=True)
    except Exception as e:
        bot_response = f"Sorry, there was an error with the local model: {str(e)}"
    
    # Add assistant message to state
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Clear the input box
    st.session_state.user_input = ''

def on_clear_click():
    """Callback to clear the chat history."""
    st.session_state.messages = []

# --- PAGE DEFINITIONS ---

def login_page():
    """Displays the login form."""
    with st.form("login_form"):
        st.title("登录")
        username = st.text_input("用户名", value="")
        password = st.text_input("密码", value="", type="password")
        submit = st.form_submit_button("登录")

        if submit:
            if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
                st.success("登录成功！")
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("用户名或密码错误，请重新输入。")

def chat_page_eyedoc(generate):
    """Displays the main chat page."""
    st.title("EyeDoc-chat Ophthalmologist")
    
    add_chat_styles()
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    # Input and clear button layout
    col1, col2 = st.columns([8, 2])
    with col1:
        st.text_input(
            "Please enter your question:",
            on_change=on_input_change,
            args=(generate,), # Pass the generate object to the callback
            key="user_input",
            placeholder="Please describe your eye symptoms or concerns...",
            label_visibility="collapsed"
        )
    with col2:
        st.button("Clear Chat", on_click=on_clear_click, use_container_width=True)

def main_page(generate):
    """Sets up the main page layout, sidebar, and page navigation."""
    st.set_page_config(
        page_title="EyeDoc-chat WebUI",
        page_icon="👁️",
        initial_sidebar_state="expanded",
        layout="wide",
        menu_items={'About': f"""Welcome to EyeDoc-chat WebUI {VERSION}!"""}
    )

    pages = {
        "EyeDoc-chat Ophthalmologist": {
            "icon": "heart-pulse",
            "func": lambda: chat_page_eyedoc(generate),
        },
    }

    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{model_avatar}" width="200" style="border-radius: 10px;">
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"""<p align="right">Current Version: {VERSION}</p>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Features")
        st.markdown("- 🩺 Professional AI Ophthalmologist Assistant")
        st.markdown("- 💬 Real-time Q&A Consultation")
        st.markdown("- 📋 Professional Medical Advice")
        st.markdown("- 🔄 Clear Chat History")
        
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        selected_page = option_menu(
            "Select Function",
            options=options,
            icons=icons,
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )
        st.session_state.selected_page = selected_page

    if selected_page in pages:
        pages[selected_page]["func"]()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    if st.session_state.logged_in:
        # Initialize the model only once after login
        if 'model_generate' not in st.session_state:
            st.session_state.model_generate = Generate(tinyllama_knowledge_param)
        
        main_page(st.session_state.model_generate)
    else:
        login_page()
