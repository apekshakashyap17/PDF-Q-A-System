# PDF Q&A System

A fully local RAG (Retrieval-Augmented Generation) system that lets you ask natural language questions about any PDF document — no external API, no internet connection required.

## What it does

Upload a PDF and ask questions about its content in plain English. The system retrieves the most relevant sections from the document and uses a locally deployed LLM to generate accurate, context-aware answers.

## How it works

1. PDF is loaded and split into chunks (`load_documents.py`, `split_documents.py`)
2. Chunks are embedded using HuggingFace embeddings (`embedding_function.py`)
3. Embeddings are stored in a ChromaDB vector store (`vector_store.py`)
4. User query is embedded and matched against stored chunks
5. Top matching chunks are passed to LLaMA 3.2 via Ollama to generate an answer (`query.py`)

## Tech Stack

- **LLM:** LLaMA 3.2 via Ollama (fully local)
- **Embeddings:** HuggingFace
- **Vector Store:** ChromaDB
- **Framework:** LangChain
- **Language:** Python

## Setup
pip install -r requirements
ollama pull llama3.2
python main.py


## Why local?

Built to run entirely on your machine — no API keys, no cost, no data leaving your device.
