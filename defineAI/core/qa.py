from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from defineAI.core.embedding import FolderIndex


# todo: ConversationalRetrievalChain internally calls the vectorstore
# so we shouldn't call the vectorstore directly to get the query result
def ask_with_memory(
    folder_index: FolderIndex,
    question: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    model: str = "gpt-3.5-turbo-16k",
    k: int = 5,
    chat_history=[],
) -> tuple[dict, list, list]:
    """
    Ask a question with memory.
    """
    if not folder_index:
        raise ValueError("folder_index is None")
    llm = ChatOpenAI(
        temperature=temperature, max_tokens=int(max_tokens), model=str(model)
    )

    retriever = folder_index.index.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    query_result = folder_index.index.similarity_search(question, k=k)

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({"question": question, "chat_history": chat_history})

    chat_history.append((question, result["answer"]))
    return result, chat_history, query_result
