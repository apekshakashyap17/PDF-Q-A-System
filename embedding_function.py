from langchain_ollama import OllamaEmbeddings
from split_documents import split_documents

def embedding_function():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )
    return embeddings


if __name__ == "__main__":
    embed = embedding_function()


    test_vector = embed.embed_query("This is a test to see my first embedding.")

    print(f"Vector Length: {len(test_vector)}") 
    print(f"First 10 numbers: {test_vector[:10]}")

    print("Embedding function executed successfully")