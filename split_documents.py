from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_documents import document_loader

def split_documents():
    documents = document_loader(r"D:\AI projects'\RAG pdf based Q&A\docs")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    print(f"Splitting documents into smaller chunks...")
    chunks = split_documents()
    print(f"Chunks created: {len(chunks)}")
    print(f"Sample chunk: {chunks[0].page_content[:]}...")