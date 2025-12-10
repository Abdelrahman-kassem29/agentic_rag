from langchain_openai import ChatOpenAI

def decompose_query(query: str):
    llm = ChatOpenAI(temperature=0)

    prompt = f"""
Decompose the following question into minimal logical sub-questions
needed to answer it correctly.

Question: {query}

Return each sub-question on a new line.
If the question is simple, return it as a single line.
"""

    response = llm.invoke(prompt).content

    sub_questions = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
    return sub_questions
