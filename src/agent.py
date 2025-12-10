from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from .retrieval import get_retriever, load_vector_store


class AgenticRAGAgent:
    """
    Simple agentic RAG agent:
    - Decomposes the user query into sub-questions (planning)
    - Retrieves documents for each sub-question (multi-step retrieval)
    - Synthesizes a final answer using the combined context
    """
    def __init__(self, vectorstore_path: str = "vectorstore", k: int = 3):
        # Load vector store and retriever
        vectorstore = load_vector_store(vectorstore_path)
        self.retriever = get_retriever(vectorstore, k=k)

        # LLM used for both planning and answering
        self.llm = ChatOpenAI(temperature=0)

    def _decompose_query(self, query: str):
        """
        Use the LLM to decompose a complex query into sub-questions.
        If the query is simple, it will just return one sub-question.
        """
        prompt = f"""
You are a helpful planner for a Retrieval-Augmented Generation (RAG) system.

Decompose the following user question into the minimal set of sub-questions
needed to answer it correctly.

- If the question is simple, just return it as a single line.
- Return each sub-question on a separate line.
- Do not add any explanations, only the sub-questions.

User question:
{query}
"""

        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        sub_questions = [
            line.strip("-• ").strip()
            for line in text.split("\n")
            if line.strip()
        ]

        # Fallback: if something went wrong, just use the original query
        if not sub_questions:
            sub_questions = [query]

        return sub_questions

    def _run_agentic_rag(self, query: str):
        """
        Full agentic pipeline:
        1. Plan: decompose the query.
        2. Retrieve: get docs for each sub-question.
        3. Answer: synthesize a final answer from the combined context.
        """
        steps = []

        # 1) Planning
        sub_questions = self._decompose_query(query)
        steps.append({"step": "planning", "sub_questions": sub_questions})

        # 2) Multi-step retrieval
        all_docs = []
        for sub_q in sub_questions:
            docs = self.retriever.get_relevant_documents(sub_q)
            all_docs.extend(docs)
            steps.append({
                "step": "retrieval",
                "sub_question": sub_q,
                "num_docs": len(docs),
            })

        # Deduplicate documents by content
        seen = set()
        unique_contents = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_contents.append(doc.page_content)

        context = "\n\n".join(unique_contents) if unique_contents else "No context retrieved."

        # 3) Final answer synthesis
        final_prompt = f"""
You are an assistant in a Retrieval-Augmented Generation (RAG) system.

You will receive:
- A user question.
- A context made of retrieved documents.

Answer the question using ONLY the information in the context.
If the context does not contain enough information, say you do not know.

Context:
{context}

User question:
{query}
"""

        answer_response = self.llm.invoke(final_prompt)
        answer_text = (
            answer_response.content
            if hasattr(answer_response, "content")
            else str(answer_response)
        )

        steps.append({"step": "answer", "answer": answer_text})

        # Return both answer and steps (useful to show "agentic" behavior)
        return {
            "answer": answer_text,
            "steps": steps,
        }

    def invoke(self, inputs):
        """
        Make this compatible with your existing `run_agent_query`,
        which calls: agent.invoke({"messages": [{"role": "user", "content": query}]})
        """
        # Support both direct string and LangChain-style dict
        if isinstance(inputs, str):
            query = inputs
        elif isinstance(inputs, dict) and "messages" in inputs:
            messages = inputs["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                query = messages[-1]["content"]
            else:
                raise ValueError("No messages found in input to agent.")
        else:
            raise ValueError(
                "AgenticRAGAgent.invoke expects a string or a dict with 'messages'."
            )

        return self._run_agentic_rag(query)


def create_rag_agent(vectorstore_path: str = "vectorstore"):
    """
    Factory used by main.py to create the agent.
    """
    return AgenticRAGAgent(vectorstore_path=vectorstore_path)


def run_agent_query(agent, query: str):
    """
    Wrapper used by main.py.
    """
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return response
