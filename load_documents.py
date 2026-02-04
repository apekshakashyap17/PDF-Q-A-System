from langchain_community.document_loaders import PyPDFDirectoryLoader


def document_loader(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()

    filtered_documents = [
        doc for doc in documents 
        if doc.metadata.get("page", 0) >= 6
    ]

    return filtered_documents


if __name__ == "__main__":   
    directory_path = r"D:\AI projects'\RAG pdf based Q&A\docs"
    documents = document_loader(directory_path)

    print(f"Loading documents from directory:{directory_path}")
    print(len(documents))
    print(documents[0])