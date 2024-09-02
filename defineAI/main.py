import streamlit as st
from dotenv import find_dotenv, load_dotenv
import os
from pathlib import Path
import sys
import pandas as pd
from pypdf import PdfReader
import io
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from defineAI.core.qa import ask_with_memory
from defineAI.utils.ui import display_metadata

current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
sys.path.append(str(parent_directory))

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))


def clear_session_state():
    keys_to_keep = ["user_input"]
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]


def normal_chatbot_sidebar():
    with st.sidebar:
        st.header("Chatbot Settings")
        model = st.selectbox(
            "Model", ["gpt-4o", "gpt-4-mini", "claude-3-5-sonnet-20240620"], index=0
        )
        api_key = (
            st.text_input("Anthropic API Key", type="password")
            if model == "claude-3-5-sonnet-20240620"
            else st.text_input("OpenAI API Key", type="password")
        )
        if api_key and model != "claude-3-5-sonnet-20240620":
            os.environ["OPENAI_API_KEY"] = api_key
        elif api_key and model == "claude-3-5-sonnet-20240620":
            os.environ["ANTHROPIC_API_KEY"] = api_key
        temperature = st.number_input(
            "Temperature", value=1.0, min_value=0.0, max_value=2.0
        )
        max_tokens = st.number_input(
            "Max tokens", value=4000, min_value=100, max_value=8000
        )
        return api_key, model, temperature, max_tokens


def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()
        return text_content
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        return df.to_string()
    else:
        return None


def display_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def ask_chatbot(model, temperature, max_tokens, messages):
    print(messages)
    parser = StrOutputParser()
    if model == "claude-3-5-sonnet-20240620":
        chatbot = ChatAnthropic(model=model)
    else:
        chatbot = ChatOpenAI(model=model)
    output = chatbot.invoke(messages, temperature=temperature, max_tokens=max_tokens)
    return parser.invoke(output)


load_dotenv(find_dotenv(), override=True)
st.title("defineAI")

api_key, model, temperature, max_tokens = normal_chatbot_sidebar()

# Dynamic system prompt input
system_message = st.text_area(
    "System Message",
    value="あなたは優秀なアシスタントです．",
    height=100,
    help="Enter the system prompt for the chatbot. This sets the context and behavior for the AI.",
)

if (
    "langchain_messages" not in st.session_state
    or st.session_state.get("system_message", "") != system_message
):
    st.session_state.langchain_messages = [SystemMessage(content=system_message)]
    st.session_state.system_message = system_message

display_chat_history()

uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

if uploaded_file and "file_content" not in st.session_state:
    file_content = process_uploaded_file(uploaded_file)
    if file_content:
        st.success(f"File '{uploaded_file.name}' processed successfully!")
        st.session_state.file_content = file_content
        context_message = (
            f"Here's the content of the uploaded file:\n\n{file_content[:2000]}..."
        )
        st.session_state.langchain_messages.append(
            SystemMessage(content=context_message)
        )

q = st.text_input("Ask a question", key="user_input")

if st.button("Submit"):
    if q:
        user_message = {"role": "user", "content": q}
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(q)

        if "file_content" in st.session_state:
            q_with_context = f"Based on the uploaded file content, please answer the following question: {q}"
        else:
            q_with_context = q

        st.session_state.langchain_messages.append(HumanMessage(content=q_with_context))

        answer = ask_chatbot(
            model, temperature, max_tokens, st.session_state.langchain_messages
        )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.langchain_messages.append(AIMessage(content=answer))

        if "file_content" in st.session_state:
            del st.session_state.file_content

        st.rerun()

if st.button("Clear Chat"):
    clear_session_state()
    st.rerun()
