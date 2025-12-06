import os
from dotenv import load_dotenv
from .ingestion import ingest_data
from .agent import create_rag_agent, run_agent_query

def main():
    # Load environment variables
    load_dotenv()

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY in the .env file.")
        return

    # Ingest data and create vector store
    print("Ingesting documents...")
    vectorstore = ingest_data("data")
    if vectorstore is None:
        print("Failed to create vector store.")
        return

    # Create the RAG agent
    print("Creating RAG agent...")
    agent = create_rag_agent()

    # Example query
    query = "What is agentic RAG?"
    print(f"Query: {query}")
    response = run_agent_query(agent, query)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
