# Baymax T&C

# 🚀 Baymax T&C: Your Friendly Terms of Service Summarizer

This repository contains the code for Baymax T&C, an LLM application designed to simplify complex Terms of Service (ToS) agreements.  We use Retrieval Augmented Generation (RAG) and large language models (LLMs) to provide clear and concise explanations of legal jargon, making ToS documents easier to understand.


## ℹ️ About Baymax T&C

Baymax T&C uses a powerful LLM and a large repository of ToS agreements. When you ask a question about a specific ToS, the application searches its document store for relevant information (using RAG). This information is then processed by the LLM, which translates the legal language into plain English and generates a concise explanation.

**Who can use Baymax T&C?**  Anyone! Consumers, employees, students, and anyone needing to understand online terms and conditions.

## ⚙️ Key Features

* **User-Friendly Interface:** Simple and intuitive.
* **Comprehensive Coverage:**  Covers a wide range of ToS agreements (currently includes: HireVue, Llama2, Mettler Toledo, Open AI, Pentair, Qdoba Mexican Eats, Bank of America, Ulta Beauty, Verizon, Truist - constantly expanding!).
* **Accurate and Reliable:**  Powered by an LLM trained on a vast dataset of legal documents.
* **Customizable:** Users can adjust the level of detail in the explanations.


## 🎯 Addressing Common ToS Problems

### Baymax T&C tackles the challenges of:

* **Legal jargon:** Complex legal language is translated into plain English.
* **Length and complexity:**  Long and complicated ToS documents are summarized.
* **Lack of clarity:**  Important terms and concepts are clearly explained.

# 🚀 RAG Evaluation and Implementation 

This repository contains a comprehensive system for building and evaluating Retrieval Augmented Generation (RAG) pipelines.  It leverages Langchain, OpenAI, FAISS, and deepeval to provide a robust and efficient solution.

## 📁 Project Overview

This project offers a complete workflow for creating a question-answering system that leverages external documents. The key components are:

1. **📄 Document Processing (helper_functions.py):**  Handles the ingestion and processing of PDF documents and text files. This includes:
    * PDF/Text Encoding: Processes both PDFs and text strings.
    * Chunking: Splits text into overlapping chunks for efficient embedding.
    * Embedding: Generates OpenAI embeddings for each chunk.
    * Vector Store: Stores embeddings in a FAISS vector database for fast similarity search.  This allows for efficient retrieval of relevant information.

2. **⚙️ RAG Pipeline (simple_rag.ipynb):**  Implements the core RAG pipeline using Langchain and OpenAI:
    * Document Loading: Loads documents from specified folders and creates a FAISS index.
    * Retrieval: Uses FAISS for efficient retrieval of relevant document chunks.  Includes a fallback BM25 method.
    * Answer Generation: Uses an OpenAI LLM (specified in the notebook) to generate answers based on the retrieved context.

3. **🤖 RAG Evaluation (evaluate_rag.py):**  Provides a thorough evaluation of the RAG system using the deepeval library. Metrics include:
    * Correctness (GEval)
    * Faithfulness
    * Answer Relevancy

## 💪 Technical Details

This project uses:

* **Python 3.10.12:** Our runtime environment.
* **Langchain:** For building the RAG pipeline.
* **OpenAI API:**  Currently used for LLM processing (initially planned to use a local LLaMA 3 model via Modal, but encountering challenges).
* **FAISS:** For efficient vector search within the document store.
* **Deepeval:** For evaluating model performance (using Correctness, Faithfulness, and Contextual Relevancy metrics).


## 🛠️ Code Quality Tools

We utilize:

* **Black:** For code formatting.
* **isort:** For import sorting.
* **Bandit:** For security analysis.
* **Pre-commit:** To automate code quality checks before each commit.


## 🐍 Setup

Before running the code, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install openai langchain faiss deepeval python-dotenv
```
Markdown
You will also need an OpenAI API key. Create a .env file in the root directory and add your key:

```bash
OPENAI_API_KEY=your_api_key_here
```

# 🏃 Usage
📂 Prepare your data: Place your PDF documents and text files in the data folder for simple_rag.ipynb. You'll likely need to modify the folders variable in that notebook to match your file structure.

▶️ Run the Jupyter Notebook: Execute simple_rag.ipynb. This will load your documents, create the vector database, and run the evaluation.

📊 Review the Results: The notebook will output the evaluation results from evaluate_rag.py, providing metrics for correctness, faithfulness, and relevance.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

📝 License

 Copyrights of the Baymax T&C Team

Happy RAG-ing! 🎉
