from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from embedding_function import embedding_function


CHROMA_PATH = "chroma_db"


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
if you do not know the answer write "I dont have enough information to answer this question "
Answer the question based on the above context: {question}
"""


def query_rag(question):
    embedding = embedding_function()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    results = db.similarity_search(question, k=5)

    context = "\n\n".join([doc.page_content for doc in results])

    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    
    llm = ChatOllama(model="llama3.2")
    response = llm.invoke(formatted_prompt)

    
    print("\nAnswer:")
    print(response.content)

    return response.content


if __name__ == "__main__":
    user_question = input("Ask a question: ")
    query_rag(user_question)
