from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings


class CodeVectorStore:
    """
    Embeds and persists Python code chunks in a Chroma vector store.

    On first run the collection is built from scratch and written to disk.
    On subsequent runs the existing collection is loaded from disk automatically,
    so documents are indexed only once.  Call `add_documents` to refresh with
    new or updated chunks.
    """

    _COLLECTION = "python_code"
    # Lightweight model that ships with sentence-transformers and works well
    # for code similarity.  Swap for e.g. "nomic-ai/nomic-embed-text-v1" if
    # you want a larger, code-oriented model.
    _EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, persist_directory: str = ".chroma") -> None:
        embeddings = HuggingFaceEmbeddings(model_name=self._EMBEDDING_MODEL)
        self._store = Chroma(
            collection_name=self._COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Index a list of (split) documents.  Safe to call multiple times."""
        self._store.add_documents(documents)

    def as_retriever(self, k: int = 5) -> VectorStoreRetriever:
        """Return a retriever that fetches the *k* most similar chunks."""
        return self._store.as_retriever(search_kwargs={"k": k})
