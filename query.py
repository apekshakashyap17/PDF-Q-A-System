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
    # 1️⃣ Load embedding model (same one used while storing data)
    embedding = embedding_function()

    # 2️⃣ Load existing Chroma vector database from disk
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    # 3️⃣ Search the database for relevant chunks
    results = db.similarity_search(question, k=5)

    # 4️⃣ Combine all retrieved chunks into one context string
    context = "\n\n".join([doc.page_content for doc in results])

    # 5️⃣ Create prompt with context + user question
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    # 6️⃣ Send prompt to LLM
    llm = ChatOllama(model="llama3.2")
    response = llm.invoke(formatted_prompt)

    # 7️⃣ Print answer
    print("\nAnswer:")
    print(response.content)

    return response.content


if __name__ == "__main__":
    user_question = input("Ask a question: ")
    query_rag(user_question)
