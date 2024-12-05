import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import toml
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import OpenAI

config_path = os.path.join(os.path.dirname(__file__), ".streamlit", "config.toml")

# Load the configuration settings
config = toml.load(config_path)

run_on_save = config["server"]["runOnSave"]
theme_base = config["theme"]["base"]
primary_color = config["theme"]["primaryColor"]

# Import functions directly from core module
from core import (
    create_retriever,
    initialize_rag,
    initialize_vectorstore_with_metadata,
    retrieve_context_per_question,
    show_context,
    text_wrap,
)

temperature = 0.4
max_tokens = 250
top_p = 1.0
presence_penalty = 0.0
frequency_penalty = 0.0

load_dotenv()

# Get the directory of the current script
current_dir = Path(__file__).parent
image_path = current_dir / "assets" / "images" / "logo.jpg"


def load_metadata(metadata_path="frontend/data/metadata.json"):
    """
    Loads metadata from the specified JSON file.

    Args:
        metadata_path (str): Path to the metadata JSON file.

    Returns:
        list: A list of dictionaries containing metadata information.
    """
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.error(f"Failed to load metadata: {str(e)}")
        return []


def get_base64_image(image_path):
    import base64

    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


col1, col2 = st.columns([4, 1])
with col2:
    st.markdown(
        f"""
        <div style="text-align: right;">
            <img src="data:image/jpg;base64,{get_base64_image(str(image_path))}" 
                 style="width: 200px; height: 200px; object-fit: contain;">
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("Baymax T&C")

# Consolidated API key handling
st.sidebar.title("OpenAI API Configuration")

# Try to get API key from environment first, then allow user input
api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key. This will not be stored permanently.",
)

# Initialize the OpenAI client only once
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()  # Stop execution until API key is provided

# Store API key in session state if it's new or different
if "api_key" not in st.session_state or st.session_state.api_key != api_key:
    st.session_state.api_key = api_key
    # Clear the retriever if API key changes
    if "retriever" in st.session_state:
        del st.session_state.retriever

client = OpenAI(api_key=api_key)
st.sidebar.success("API key provided successfully!")

# Dropdown to display available T&Cs
metadata = load_metadata()
if metadata:
    titles = ["Browse companies"] + [
        meta["title"] for meta in metadata
    ]  # Extract only the titles
    selected_tc = st.sidebar.selectbox(
        "Available Terms and Conditions", titles, index=0, key="browse_only"
    )
    # st.markdown(f"### Selected: **{selected_tc}**")
else:
    st.sidebar.warning("No terms and conditions available.")

# Initialize RAG system after API key verification
if "retriever" not in st.session_state:
    init_message = st.empty()
    try:
        os.environ["OPENAI_API_KEY"] = api_key  # Set API key for RAG system
        st.session_state.retriever = initialize_rag()
        init_message.success("RAG system initialized successfully!")
        time.sleep(3)
        init_message.empty()
    except Exception as e:
        init_message.error(
            f"Failed to initialize RAG system. Error: {str(e)}\nContinuing without RAG functionality."
        )
        print(
            f"RAG initialization error: {str(e)}", file=sys.stderr
        )  # Detailed logging
        # Don't stop the app, just continue without RAG
        st.session_state.retriever = None

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Model selection
model_options = ["gpt-4", "gpt-4-turbo-preview"]
selected_model = st.sidebar.selectbox("Select Model", model_options)
st.session_state["openai_model"] = selected_model

# Reset button
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.sidebar.success("Chat session reset!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            formatted_content = (
                message["content"].replace("\n*", "\n\n*").replace("\n-", "\n\n-")
            )
            st.markdown(formatted_content)
        else:
            st.markdown(message["content"])

# System prompt
system_prompt = """You are a knowledgeable assistant with access to specific terms and conditions via a Retrieval-Augmented Generation (RAG) system.
- You can provide answers based on terms and conditions you have access to.
- If a question is outside your RAG system, you can answer using general knowledge.

Always prioritize context retrieved via the RAG system to ensure accuracy."""

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize variables
    context = ""
    is_metadata_query = False

    # Check if retriever exists
    if "retriever" in st.session_state and st.session_state.retriever is not None:
        try:
            # Check if it's a metadata query
            if any(
                keyword in prompt.lower()
                for keyword in [
                    "terms and conditions",
                    "available",
                    "what terms",
                    "companies",
                ]
            ):
                is_metadata_query = True
                context = retrieve_context_per_question(
                    prompt, st.session_state.retriever
                )

                # Display metadata titles in the chat
                with st.chat_message("assistant"):
                    if context:
                        st.write("### Available Terms and Conditions:")
                        for title in context:
                            st.write(f"- {title}")
                    else:
                        st.write("No terms and conditions available.")

            # Retrieve context for general queries
            if not is_metadata_query:
                context = retrieve_context_per_question(
                    prompt, st.session_state.retriever
                )

        except Exception as e:
            st.warning(f"RAG retrieval error: {str(e)}")

    # Handle general queries or continue if context exists
    if not is_metadata_query:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Include context in the prompt if available
                system_prompt_with_context = system_prompt
                if context:
                    if isinstance(context, list):
                        context_str = "\n\n".join(context)
                    else:
                        context_str = context
                    system_prompt_with_context = (
                        f"{system_prompt}\n\nRelevant context:\n{context_str}"
                    )

                # Generate assistant response
                messages = [{"role": "system", "content": system_prompt_with_context}]
                messages.extend(
                    [
                        {"role": str(m["role"]), "content": str(m["content"])}
                        for m in st.session_state.messages
                    ]
                )

                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                # Format final response
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                full_response = "I apologize, but I encountered an error while processing your request."

            # Save assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
