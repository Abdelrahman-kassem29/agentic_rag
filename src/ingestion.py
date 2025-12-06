import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def load_documents(data_dir: str):
    """Load documents from the data directory."""
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

def create_vector_store(documents, persist_directory: str = "vectorstore"):
    """Create and persist a FAISS vector store from documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Persist the vector store
    vectorstore.save_local(persist_directory)
    return vectorstore

def ingest_data(data_dir: str, persist_directory: str = "vectorstore"):
    """Main function to ingest data and create vector store."""
    documents = load_documents(data_dir)
    if not documents:
        print("No documents found in the data directory.")
        return None

    vectorstore = create_vector_store(documents, persist_directory)
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore
