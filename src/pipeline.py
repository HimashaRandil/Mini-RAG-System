# src/utils/pipeline.py

import pandas as pd
import chromadb
from fastembed import TextEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logger.logging import logger as logger


def load_and_preprocess_data(filepath, num_rows=300):
    """
    Loads a subset of the dataset and prepares it for processing.
    """
    logger.info(f"Loading and preprocessing {num_rows} rows from {filepath}...")
    df = pd.read_csv(filepath)
    df = df.head(num_rows)
    df.dropna(subset=["Plot"], inplace=True)

    documents = [
        f"Title: {row['Title']}\nPlot: {row['Plot']}" for _, row in df.iterrows()
    ]
    logger.info(f"Loaded {len(documents)} documents.")
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits the documents into smaller chunks for embedding.
    """
    logger.info(f"Chunking {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_docs = text_splitter.create_documents(documents)
    chunks = [doc.page_content for doc in chunked_docs]
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks):
    """
    Embeds chunks and stores them in an in-memory ChromaDB collection.
    """
    logger.info("Initializing embedding model and vector store...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embeddings = list(embedding_model.embed(chunks))

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="movie_plots")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)

    logger.info("Vector store created successfully.")
    return collection, embedding_model
