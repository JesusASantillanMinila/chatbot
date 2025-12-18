import streamlit as st
import tempfile
import os
import json

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_community import GoogleDriveLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page Configuration
st.set_page_config(page_title="Professional Bot", layout="wide")
st.markdown('<h1>Minil.Ai</h1>', unsafe_allow_html=True)
li_url = "https://www.linkedin.com/in/jesussantillanminila/"
st.markdown(f"Hi, I am a chatbot built by [Jesus Santillan Minila]({li_url}) to answer questions about his career.")

# --- 1. Credentials Handling ---
# The GoogleDriveLoader requires a file path for the service account.
# We securely write the secret JSON to a temporary file.
if "google_credentials" in st.secrets:
    # Create a temporary file for the service account credentials
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp:
        json.dump(dict(st.secrets["google_credentials"]), temp)
        temp.flush()
        SERVICE_ACCOUNT_FILE = temp.name
else:
    st.error("Missing 'google_credentials' in Streamlit secrets.")
    st.stop()

# --- 2. Load and Index Documents (Cached) ---
@st.cache_resource(show_spinner=True)
def initialize_vector_store():
    """
    Loads docs from Drive, splits them, and creates a local FAISS vector store.
    """
    try:
        folder_id = st.secrets["DRIVE_FOLDER_ID"]
        
        # Initialize Google Drive Loader
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            service_account_key=SERVICE_ACCOUNT_FILE,
            recursive=False  # Set to True if you want to search subfolders
        )
        
        with st.spinner(f"Loading documents from Drive Folder: {folder_id}..."):
            docs = loader.load()
            
        if not docs:
            st.warning("No documents found in the specified folder.")
            return None

        # Split text into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        # Create Embeddings (Free using Gemini)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )

        # Build Vector Store
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore

    except Exception as e:
        st.error(f"Error connecting to Google Drive: {e}")
        return None

# Initialize the RAG system
if "vector_store" not in st.session_state:
    st.session_state.vector_store = initialize_vector_store()

# --- 3. Chat Logic ---

if st.session_state.vector_store:
    # Initialize LLM (Gemini 1.5 Flash is fast and free-tier eligible)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    # Create the Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based strictly on the context provided below.
    If the answer is not in the context, say "I couldn't find that information in your Google Docs."
    
    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Create Chains (Modern Approach)
    # 1. Create a document chain (handling the LLM + Prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 2. Create the retrieval chain (handling Retriever + Document Chain)
    retriever = st.session_state.vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # --- 4. UI Interface ---
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt_text := st.chat_input("Ask a question about your docs..."):
        # Display user message
        st.chat_message("user").markdown(prompt_text)
        st.session_state.messages.append({"role": "user", "content": prompt_text})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = retrieval_chain.invoke({"input": prompt_text})
                answer = response['answer']
                st.markdown(answer)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Cleanup: Remove the temporary credentials file on script exit
if os.path.exists(SERVICE_ACCOUNT_FILE):
    os.remove(SERVICE_ACCOUNT_FILE)