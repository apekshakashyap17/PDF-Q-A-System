import os 
from langchain_chroma import Chroma
from embedding_function import embedding_function
from split_documents import split_documents


CHROMA_PATH = "chroma_db"

def create_vector_store():
    embed = embedding_function()

    if os.path.exists(CHROMA_PATH):
        print("--- Loading existing database ---")
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embed
        )
    else:
        print("--- Creating NEW database ---")
        chunks = split_documents()
        db = Chroma.from_documents(
            documents=chunks, 
            embedding=embed, 
            persist_directory=CHROMA_PATH
        )
        print(f"✅ Success! Created database with {len(chunks)} chunks.")
    
    return db

if __name__ == "__main__":
    vector_db = create_vector_store()