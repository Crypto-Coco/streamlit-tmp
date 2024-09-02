from typing import List
from langchain.docstore.document import Document
import streamlit as st
import tiktoken


def get_embedding_cost(texts: List[Document]) -> tuple[int, float]:
    """
    Prints the total number of tokens and the cost of embedding the documents.
    """
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004
