from .helpers import (
    encode_pdf,
    generate_document_summary,
    process_uploaded_tc,
    read_file_content,
    retrieve_all_metadata,
    retrieve_context_per_question,
)
from .rag import (
    create_documents_with_metadata,
    create_retriever,
    encode_documents,
    initialize_rag,
    initialize_vectorstore_with_metadata,
    setup_environment,
)

# Make these functions available when importing from core
__all__ = [
    # RAG functions
    "initialize_rag",
    "create_retriever",
    "encode_documents",
    "setup_environment",
    "initialize_vectorstore_with_metadata",
    "create_documents_with_metadata",
    # Helper functions
    "encode_pdf",
    "retrieve_context_per_question",
    "retrieve_all_metadata"
    # Document upload functions
    "read_file_content",
    "generate_document_summary",
    "process_uploaded_tc",
]
