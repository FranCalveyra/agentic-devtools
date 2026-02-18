from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


class CodeSplitter:
    """
    Splits Python source documents into smaller chunks using AST-aware boundaries
    (classes, functions, etc.) provided by RecursiveCharacterTextSplitter.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        self._splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)
