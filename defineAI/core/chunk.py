from typing import List
from langchain.docstore.document import Document
from defineAI.core.parse import File
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_data(
    file: File,
    chunk_size: int = 256,
    chunk_overlap: int = 20,
    model_name="gpt-3.5-turbo-16k",
) -> File:
    """
    Chunk a file into smaller pieces.
    """
    chunked_data: List[Document] = []
    for doc in file.docs:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        chunks = text_splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata.get("page", 1),
                    "chunk": i + 1,
                    "source": f"{doc.metadata.get('page', 1)}-{i + 1}",
                },
            )
            chunked_data.append(doc)

    chunked_file = file.copy()
    chunked_file.docs = chunked_data
    return chunked_file
