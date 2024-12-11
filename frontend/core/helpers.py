import asyncio
import io
import random
import textwrap
from pathlib import Path
from typing import List, Tuple

import docx
import numpy as np
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from openai import RateLimitError
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi


def read_file_content(uploaded_file) -> str:
    """
    Read content from various file formats (PDF, DOCX, TXT).

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        str: Extracted text content from the file.
    """
    content = ""
    file_extension = Path(uploaded_file.name).suffix.lower()

    if file_extension == ".pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")

    elif file_extension == ".docx":
        try:
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {str(e)}")

    elif file_extension == ".txt":
        try:
            content = uploaded_file.getvalue().decode()
        except Exception as e:
            raise ValueError(f"Error reading TXT: {str(e)}")

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return content


def generate_document_summary(content: str, client) -> str:
    """
    Generate a summary of the uploaded T&C document using OpenAI.

    Args:
        content (str): The full text content of the document.
        client: OpenAI client instance.

    Returns:
        str: A structured summary of the document.
    """
    system_prompt = """You are a legal document analyzer specializing in Terms and Conditions analysis. 
    Provide a clear, structured summary of the document covering these key aspects:
    1. Document Overview (2-3 sentences)
    2. Key Terms and Definitions
    3. Main User Rights and Obligations
    4. Important Limitations or Restrictions
    5. Notable Clauses or Provisions
    
    Keep the summary concise but informative. Focus on the most important points that users should be aware of and do not lie."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Updated to use a valid model
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Please analyze and summarize this Terms and Conditions document:\n\n{content}",
                },
            ],
            temperature=0.5,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def process_uploaded_tc(content: str, client, session_state) -> Tuple[List[str], str]:
    """
    Process uploaded T&C content and generate a summary, with session state handling.

    Args:
        content (str): The full text content of the document.
        client: OpenAI client instance.
        session_state: Streamlit session state object.

    Returns:
        Tuple[List[str], str]: A tuple containing (chunks of text, document summary).
    """
    # Check if content is already in session state
    content_hash = hash(content)  # Create a hash of the content to use as identifier

    if "processed_documents" not in session_state:
        session_state.processed_documents = {}

    if content_hash in session_state.processed_documents:
        # Return cached results
        return session_state.processed_documents[content_hash]

    # Split content into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,  # Adjust chunk size to stay within token limits
        chunk_overlap=500,
        length_function=len,
    )
    chunks = text_splitter.split_text(content)

    # Generate a summary for each chunk
    summaries = []
    for chunk in chunks:
        try:
            summary = generate_document_summary(chunk, client)
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"Error generating summary for a chunk: {str(e)}")

    # Combine summaries into a single summary
    combined_summary = "\n\n".join(summaries)

    # Store results in session state
    session_state.processed_documents[content_hash] = (chunks, combined_summary)

    return chunks, combined_summary


def replace_t_with_space(list_of_documents):
    """
    Replace all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace("\t", " ")
    return list_of_documents


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded PDF content.
    """
    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

def retrieve_all_metadata(vectorstore):
    """
    Retrieve all unique metadata titles from the vectorstore.

    Args:
        vectorstore: The FAISS vectorstore containing documents and metadata.

    Returns:
        list: A list of unique document titles.
    """
    try:
        if hasattr(vectorstore, "docstore") and vectorstore.docstore:
            # Access all stored documents in the vectorstore
            documents = vectorstore.docstore._dict.values()

            # Extract unique titles from the metadata
            titles = {doc.metadata.get("title", "Unknown") for doc in documents}
            return sorted(titles)  # Return sorted list of titles for consistency
        else:
            raise ValueError("Vectorstore does not have a valid 'docstore' or metadata.")
    except Exception as e:
        raise ValueError(f"Metadata retrieval error: {e}")


def retrieve_context_per_question(question, retriever):
    """
    Retrieves metadata or context based on the user's query.

    Args:
        question (str): User's question.
        retriever: A retriever object.

    Returns:
        list: A list of metadata titles if the query is about terms, or context otherwise.
    """
    if any(
        keyword in question.lower()
        for keyword in [
            "what terms and conditions do you have access to",
            "what companies terms and conditions do you have access to",
        ]
    ):
        try:
            return retrieve_all_metadata(retriever.vectorstore)
        except Exception as e:
            raise ValueError(f"Error retrieving metadata: {e}")

    # Retrieve relevant context for general questions
    try:
        results = retriever.get_relevant_documents(question)
        return (
            [doc.page_content for doc in results]
            if results
            else ["No relevant context found."]
        )
    except Exception as e:
        raise ValueError(f"Error retrieving context: {e}")
