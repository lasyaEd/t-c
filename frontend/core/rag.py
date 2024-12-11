import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def setup_environment():
    """Setup environment variables and API key."""
    # Load environment variables
    load_dotenv()

    # Set OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    return False


def safe_load_metadata(metadata_path):
    """
    Safely load metadata from a JSON file.

    Args:
        metadata_path: Path to the metadata file.

    Returns:
        list: Metadata as a list of dictionaries.
    """
    if not os.path.exists(metadata_path):
        st.warning("Metadata file does not exist. Initializing with an empty list.")
        return []

    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error("Metadata file is invalid. Initializing with an empty list.")
        return []


def add_file_to_metadata(uploaded_file, metadata_path, data_folder):
    """
    Add uploaded file metadata to metadata.json and save the file to the data folder.

    Args:
        uploaded_file: The uploaded file object.
        metadata_path: Path to the metadata JSON file.
        data_folder: Path to the data folder.

    Returns:
        None
    """
    metadata = safe_load_metadata(metadata_path)

    # Save the uploaded file to the data folder
    file_path = os.path.join(data_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Add the new file's metadata
    new_metadata = {
        "title": uploaded_file.name.split(".")[0],  # Use file name (without extension) as the title
        "filename": uploaded_file.name,
    }
    metadata.append(new_metadata)

    # Write updated metadata back to the file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    st.success(f"File '{uploaded_file.name}' added to metadata!")


def create_retriever(vector_store, k=3):
    """Create a retriever from the vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})


def create_documents_with_metadata(
    metadata_list, data_folder, chunk_size=1000, chunk_overlap=200
):
    """
    Create documents with metadata and content chunks.
    """
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
    """
    Initialize a vector store using metadata and document chunks.

    Args:
        metadata_file: Path to the metadata JSON file.
        data_folder: Path to the folder containing text files.
        chunk_size: Maximum chunk size for splitting text.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        FAISS: A vector store containing embedded document chunks.
    """
    metadata_list = safe_load_metadata(metadata_file)
    documents = create_documents_with_metadata(
        metadata_list, data_folder, chunk_size, chunk_overlap
    )
    embeddings = OpenAIEmbeddings()

    # Ensure metadata is being printed for debugging
    for doc in documents:
        print(f"Document Metadata: {doc.metadata}")  # Debugging step

    return FAISS.from_documents(documents, embeddings)


@st.cache_resource
def initialize_rag(metadata_file, data_folder, k=2):
    """
    Initialize the RAG system with metadata and documents.

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
        metadata_list = safe_load_metadata(metadata_file)
        documents = []

        for meta in metadata_list:
            file_path = os.path.join(data_folder, meta["filename"])
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.create_documents([text])

                # Attach metadata to each chunk
                for chunk in chunks:
                    chunk.metadata = meta
                documents.extend(chunks)
            else:
                st.warning(f"File '{meta['filename']}' not found in data folder.")

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
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
