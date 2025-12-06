from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def load_vector_store(persist_directory: str = "vectorstore"):
    """Load a persisted FAISS vector store."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(persist_directory, embeddings)
    return vectorstore

def get_retriever(vectorstore, k: int = 3):
    """Get a retriever from the vector store."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

def retrieve_documents(query: str, retriever):
    """Retrieve relevant documents for a query."""
    docs = retriever.get_relevant_documents(query)
    return docs
