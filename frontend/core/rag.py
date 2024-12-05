import json
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def setup_environment():
    """Setup environment variables and API key"""
    # Load environment variables
    load_dotenv()

    # Set OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    return False


def create_retriever(vector_store, k=3):
    """Create a retriever from the vector store"""
    return vector_store.as_retriever(search_kwargs={"k": k})


def load_metadata(metadata_file):
    """
    Load metadata from a JSON file.

    Args:
        metadata_file (str): Path to the metadata.json file.

    Returns:
        list: A list of metadata dictionaries.
    """
    with open(metadata_file, "r") as f:
        return json.load(f)


def create_documents_with_metadata(
    metadata_list, data_folder, chunk_size=1000, chunk_overlap=200
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    documents = []

    for meta in metadata_list:
        file_path = os.path.join(data_folder, meta["filename"])
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                chunks = text_splitter.create_documents([text])
                for chunk in chunks:
                    chunk.metadata = meta  # Attach metadata
                documents.extend(chunks)
            print(f"Document '{meta['title']}' split into {len(chunks)} chunks.")
        else:
            print(f"Error: File {file_path} does not exist for metadata: {meta}")
            raise FileNotFoundError(f"File {file_path} not found.")
    return documents


def initialize_vectorstore_with_metadata(
    metadata_file, data_folder, chunk_size=1000, chunk_overlap=200
):
    metadata_list = load_metadata(metadata_file)
    documents = create_documents_with_metadata(
        metadata_list, data_folder, chunk_size, chunk_overlap
    )
    embeddings = OpenAIEmbeddings()

    # Ensure metadata is being printed for debugging
    for doc in documents:
        print(f"Document Metadata: {doc.metadata}")  # Debugging step

    return FAISS.from_documents(documents, embeddings)


@st.cache_resource
def initialize_rag(
    metadata_file="frontend/data/metadata.json", data_folder="frontend/data/", k=2
):
    """
    Initialize the RAG system with metadata and context.

    Args:
        metadata_file (str): Path to the metadata file.
        data_folder (str): Path to the folder containing text files.
        k (int): Number of documents to retrieve.

    Returns:
        retriever: A retriever initialized with metadata and documents.
    """
    if not setup_environment():
        raise ValueError("Failed to load API key")

    try:
        vectorstore = initialize_vectorstore_with_metadata(metadata_file, data_folder)
        return vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        raise ValueError(f"Failed to initialize RAG system: {e}")


def encode_documents(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes all text files into a vector store using OpenAI embeddings and includes metadata.

    Args:
        path: The path to the directory of text files.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded content and metadata of the files.
    """
    metadata_path = os.path.join(path, "metadata.json")
    with open(metadata_path, "r") as meta_file:
        metadata_list = json.load(meta_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    embeddings = OpenAIEmbeddings()

    documents_with_metadata = []

    for meta in metadata_list:
        file_path = os.path.join(path, meta["filename"])

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Split the text into chunks
            chunks = text_splitter.create_documents([text])

            # Attach metadata to each chunk
            for chunk in chunks:
                chunk.metadata = meta  # Attach metadata
                print(f"Document Metadata: {chunk.metadata}")  # Debug log
            documents_with_metadata.extend(chunks)
        else:
            raise FileNotFoundError(f"File {file_path} not found.")

    # Create vector store with metadata
    vectorstore = FAISS.from_documents(documents_with_metadata, embeddings)
    print("Vectorstore successfully created with metadata.")
    return vectorstore
