import streamlit as st
import os
from langchain.vectorstores import Chroma
from defineAI.core.chunk import chunk_data
from defineAI.core.embedding import (
    delete_all_embeddings,
    embed_files,
    show_loaded_documents,
)
from defineAI.core.parse import read_file
from defineAI.utils.utils import get_embedding_cost


def clear_vector_store():
    if "folder_index" in st.session_state:
        vector_db = Chroma()
        delete_all_embeddings(vector_db)
    if "uploaded_files" in st.session_state:
        del st.session_state.uploaded_files


def rag_sidebar():
    with st.sidebar:
        st.header("RAG Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        model = st.selectbox("model", ["gpt-3.5-turbo-16k", "gpt-4", "claude"], index=0)
        temperature = st.number_input(
            "temperature", value=1.0, min_value=0.1, max_value=2.0
        )
        max_tokens = st.number_input(
            "max tokens", value=4000, min_value=100, max_value=8000
        )
        uploaded_file = st.file_uploader("upload a file: ", type=["pdf", "docx", "txt"])
        is_chunk = st.checkbox("chunk data", value=True)
        chunk_size = 256
        if is_chunk:
            chunk_size = st.number_input(
                "chunk size", value=256, min_value=100, max_value=4096
            )
        k = st.number_input("k", value=5, min_value=1, max_value=10)

        cols = st.columns([3, 4, 2])
        with cols[0]:
            add_data = st.button("add data")
        with cols[1]:
            delete_data = st.button("delete data", on_click=clear_vector_store)

        cols = st.columns([2, 1, 1])
        with cols[0]:
            check_data = st.button("check data")

        return (
            api_key,
            model,
            temperature,
            max_tokens,
            uploaded_file,
            is_chunk,
            chunk_size,
            k,
            add_data,
            check_data,
        )


def process_rag(
    api_key,
    model,
    temperature,
    max_tokens,
    uploaded_file,
    is_chunk,
    chunk_size,
    k,
    add_data,
    check_data,
):
    if check_data:
        vector_db = Chroma()
        show_loaded_documents(vector_db)

    if uploaded_file and add_data:
        chunks = None
        with st.spinner("loading document..."):
            if "uploaded_files" not in st.session_state:
                st.session_state.uploaded_files = []
            st.session_state.uploaded_files.append(uploaded_file.name)

            file = read_file(uploaded_file)
            if is_chunk:
                chunks = chunk_data(file, chunk_size=int(chunk_size))
            else:
                chunks = file

            folder_index = embed_files(
                files=[chunks],
                embedding="openai",
                vector_store="chroma",
                openai_api_key=api_key,
            )
            st.session_state.folder_index = folder_index
            st.success(f"{uploaded_file.name} added successfully")

        st.write(f"number of chunks: {len(chunks.docs)}")
        total_tokens, cost = get_embedding_cost(chunks.docs)
        st.write(f"total tokens: {total_tokens}")
        st.write(f'cost: ${"{:.4f}".format(cost)}')
