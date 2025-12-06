# Agentic RAG Project

An intelligent Retrieval-Augmented Generation (RAG) system with agentic capabilities that combines document ingestion, vector search, and autonomous agents for advanced question answering and task completion.

## Features

- **Document Ingestion**: Automatically load and process documents from a specified directory
- **Vector Storage**: Create and persist FAISS vector stores using OpenAI embeddings
- **Intelligent Retrieval**: Retrieve relevant documents based on semantic similarity
- **Agentic RAG**: Utilize LangChain agents for orchestrating complex tasks and decision-making
- **OpenAI Integration**: Leverage OpenAI's GPT models for both embeddings and language generation

## Project Structure

```
agentic_rag_project/
├── src/
│   ├── main.py          # Main entry point orchestrating the RAG pipeline
│   ├── ingestion.py     # Document loading and vector store creation
│   ├── retrieval.py     # Vector store management and document retrieval
│   └── agent.py         # RAG agent creation and execution
├── data/
│   ├── sample.txt       # Sample document for testing
│   └── eval_examples.csv # Evaluation examples
├── requirements.txt     # Python dependencies
├── TODO.md             # Project tasks and progress
└── README.md           # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agentic_rag_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Complete Pipeline

Execute the main script to run the full RAG pipeline:

```bash
python -m src.main
```

This will:
1. Load environment variables
2. Ingest documents from the `data/` directory
3. Create/update the FAISS vector store
4. Initialize the RAG agent
5. Run a sample query: "What is agentic RAG?"

### Custom Queries

To run custom queries, modify the `query` variable in `src/main.py` or extend the script to accept command-line arguments.

## Key Components

### Document Ingestion (`src/ingestion.py`)
- Loads text documents from the `data/` directory
- Splits documents into chunks for optimal retrieval
- Creates FAISS vector store with OpenAI embeddings

### Retrieval System (`src/retrieval.py`)
- Loads persisted vector stores
- Provides retriever functionality for semantic search
- Returns relevant documents based on query similarity

### Agent Framework (`src/agent.py`)
- Creates a ReAct agent with RAG capabilities
- Integrates retrieval tools for knowledge-based responses
- Handles complex multi-step reasoning tasks

## Dependencies

- `langchain>=0.1.0`: Core framework for LLM applications
- `langchain-openai>=0.1.0`: OpenAI integrations
- `langchain-community>=0.0.10`: Community-contributed components
- `faiss-cpu>=1.7.4`: Vector similarity search
- `python-dotenv>=1.0.0`: Environment variable management
- `pypdf>=3.17.4`: PDF document processing
- `unstructured>=0.10.30`: Unstructured document loading
- `tiktoken>=0.5.1`: Tokenization utilities

## Data Format

The system currently supports text files (`.txt`) in the `data/` directory. Documents are automatically loaded, chunked, and indexed for retrieval.

## Evaluation

Evaluation examples are provided in `data/eval_examples.csv` for testing the system's performance on various queries.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Future Enhancements

- Support for additional document formats (PDF, DOCX, etc.)
- Web interface for query interaction
- Multi-agent architectures
- Fine-tuned models for domain-specific tasks
- Performance benchmarking tools
