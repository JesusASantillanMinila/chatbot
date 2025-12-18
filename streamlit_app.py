import streamlit as st
import os
import tempfile
import json
from langchain_google_community import GoogleDriveLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. PAGE CONFIG
st.set_page_config(page_title="Google Drive RAG Bot", layout="wide")
st.title("🤖 Chat with your Google Drive Docs")

# 2. SECRETS & CREDENTIALS SETUP
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing GOOGLE_API_KEY in secrets.")
    st.stop()

if "DRIVE_FOLDER_ID" not in st.secrets:
    st.error("Missing DRIVE_FOLDER_ID in secrets.")
    st.stop()

if "service_account" not in st.secrets:
    st.error("Missing service_account json in secrets.")
    st.stop()

# Set the API Key for LangChain Google GenAI
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Helper: Write service account dict to a temporary file for the Loader
@st.cache_resource
def get_service_account_file():
    """Writes the service account secret to a temp file and returns the path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
        return f.name

# 3. LOAD DOCUMENTS & CREATE VECTOR STORE (Cached)
@st.cache_resource
def initialize_vector_store():
    """
    Loads docs from Google Drive and creates a FAISS vector store.
    This function is cached so it doesn't run on every interaction.
    """
    temp_creds_path = get_service_account_file()
    folder_id = st.secrets["DRIVE_FOLDER_ID"]
    
    with st.spinner(f"Loading documents from Drive Folder: {folder_id}..."):
        # Load documents
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            service_account_key=temp_creds_path,
            recursive=False  # Set to True if you want subfolders
        )
        docs = loader.load()
        
        if not docs:
            st.warning("No documents found in the specified folder.")
            return None

        # Split text (optional but recommended for larger docs)
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create Vector Store using Google's Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        return vectorstore

# Initialize the knowledge base
vector_store = initialize_vector_store()

if vector_store is None:
    st.stop()

# 4. SETUP RAG CHAIN
# Create the retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Setup the LLM (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    max_tokens=None, 
    timeout=None
)

# Create the prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Build the chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt_text := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt_text})
            answer = response["answer"]
            st.markdown(answer)
            
            # Optional: Show source documents in an expander
            with st.expander("View Source Documents"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                    st.text(doc.page_content[:200] + "...")

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})