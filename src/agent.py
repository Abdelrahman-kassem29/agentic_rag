from langchain.agents import create_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from .retrieval import get_retriever, load_vector_store

def create_rag_agent(vectorstore_path: str = "vectorstore"):
    """Create an agent with RAG capabilities."""
    # Load vector store
    vectorstore = load_vector_store(vectorstore_path)

    # Create retriever
    retriever = get_retriever(vectorstore)

    # Create RAG chain
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Define tools for the agent
    tools = [
        Tool(
            name="KnowledgeBase",
            func=qa_chain.run,
            description="Useful for answering questions based on the knowledge base. Input should be a question."
        )
    ]

    # Create the agent
    agent = create_agent(llm, tools)

    return agent

def run_agent_query(agent, query: str):
    """Run a query through the agent."""
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return response
