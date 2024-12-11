import json
import os
from pathlib import Path
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI
from core import (
    create_retriever,
    process_uploaded_tc,
    read_file_content,
    retrieve_context_per_question,
)

# Load environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
if not api_key:
    st.sidebar.warning("Please provide your OpenAI API key in the sidebar.")
    st.stop()

client = OpenAI(api_key=api_key)

# Define system prompt for ToS generation
system_prompt = """
You are a legal assistant. Based on the provided context, generate a terms of service template.
Include the disclaimer:
"This template is based on existing terms and conditions. Customize it to your needs and consult a legal expert."
"""

# Tabs for functionalities
tab1, tab2, tab3 = st.tabs(["Upload T&Cs", "Browse Available T&Cs", "Generate ToS Template"])

# Tab 1: Upload T&Cs
with tab1:
    st.header("Upload Terms and Conditions")
    uploaded_file = st.file_uploader(
        "Upload a .txt file containing T&Cs", type=["txt"], help="Supported format: .txt"
    )

    if uploaded_file:
        try:
            # Read file content
            content = read_file_content(uploaded_file)

            # Process content to generate summary
            chunks, summary = process_uploaded_tc(content, client, st.session_state)

            # Display the summary
            st.subheader("Summary of Uploaded T&Cs")
            st.markdown(summary)

            # Highlight risks/unusual clauses
            st.subheader("Important Things to Keep in Mind")
            unusual_clauses = [
                "liability waiver",
                "data sharing without consent",
                "automatic renewals"
            ]  # Example risk-related phrases
            risks = [chunk for chunk in chunks if any(phrase in chunk.lower() for phrase in unusual_clauses)]
            if risks:
                for risk in risks:
                    st.warning(risk)
            else:
                st.info("No unusual clauses detected.")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Tab 2: Browse Available T&Cs
with tab2:
    st.header("Available Terms and Conditions")
    # Retrieve metadata from RAG
    retriever = st.session_state.get("retriever")
    if retriever:
        available_terms = retrieve_context_per_question("what companies terms and conditions do you have access to", retriever)
        if available_terms:
            selected_tc = st.selectbox("Select a T&C to view", available_terms)
            if selected_tc:
                # Retrieve and display summary
                context = retrieve_context_per_question(selected_tc, retriever)
                st.subheader(f"Summary of {selected_tc}")
                st.markdown(context)
        else:
            st.warning("No terms and conditions available in the database.")
    else:
        st.warning("No retriever initialized. Upload T&Cs to build the database.")

# Tab 3: Generate ToS Template
with tab3:
    st.header("Generate Terms of Service Template")
    user_request = st.text_input(
        "Enter details for the ToS template:",
        placeholder="e.g., Generate a ToS for an automobile company based on Audi's and BMW's T&Cs.",
    )

    if st.button("Generate Template"):
        if not user_request:
            st.warning("Please enter details to generate a ToS template.")
        else:
            try:
                # Retrieve context from RAG
                context = retrieve_context_per_question(user_request, retriever)
                if not context:
                    st.warning("No relevant terms found in the database. Please upload a file.")
                    uploaded_file = st.file_uploader(
                        "Upload a .txt file for ToS template generation", type=["txt"]
                    )
                    if uploaded_file:
                        content = read_file_content(uploaded_file)
                        chunks, _ = process_uploaded_tc(content, client, st.session_state)
                        context = " ".join(chunks)

                if context:
                    # Generate ToS template
                    system_prompt_with_context = f"{system_prompt}\n\nRelevant context:\n{context}"
                    response = client.chat.completions.create(
                        model="gpt-4",  # Update with your model
                        messages=[
                            {"role": "system", "content": system_prompt_with_context},
                            {"role": "user", "content": user_request},
                        ],
                        temperature=0.4,
                        max_tokens=500,
                    )
                    st.subheader("Generated Terms of Service Template")
                    st.text(response.choices[0].message["content"])
                else:
                    st.error("No context provided. Unable to generate a template.")
            except Exception as e:
                st.error(f"Error generating template: {str(e)}")
