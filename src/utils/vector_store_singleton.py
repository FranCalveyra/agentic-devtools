# Lazy singleton — HuggingFace model is loaded only when the vector store is first used.


def get_vector_store():
    """Return the shared CodeVectorStore singleton, creating it on first call."""
    global vector_store
    if vector_store is None:
        from rag.vector_store import CodeVectorStore

        vector_store = CodeVectorStore()
    return vector_store
