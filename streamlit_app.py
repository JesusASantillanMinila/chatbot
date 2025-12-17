import streamlit as st

# --- Google / Drive Imports ---
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- LangChain Core Imports ---
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# --- LangChain Google GenAI Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- LangChain Community & Splitters ---
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LangChain Chains (Modern 0.3.x path) ---
from langchain.chains.combine_documents import create_stuff_documents_chain

import io
import time

# --- Page Config ---
st.set_page_config(page_title="Professional Bot", layout="wide")


# --- 1. Drive Authentication & Fetching (Cached) ---

def get_drive_service():
    """Authenticates with Google Drive using Streamlit Secrets."""
    creds_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=creds)

def get_folder_text(folder_id):
    """Downloads and reads all PDFs/Docs from the secret folder."""
    service = get_drive_service()
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get('files', [])
    
    all_text = ""
    if not files:
        return None

    # Visual feedback only during the initial load
    progress_bar = st.progress(0, text="Connecting to Google Drive...")
    
    for index, file in enumerate(files):
        file_id = file['id']
        name = file['name']
        mime = file['mimeType']
        
        progress_bar.progress((index + 1) / len(files), text=f"Reading file: {name}")

        try:
            if mime == 'application/pdf':
                request = service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    _, done = downloader.next_chunk()
                
                fh.seek(0)
                pdf_reader = PdfReader(fh)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"

            elif mime == 'application/vnd.google-apps.document':
                request = service.files().export_media(fileId=file_id, mimeType='text/plain')
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    _, done = downloader.next_chunk()
                all_text += fh.getvalue().decode('utf-8') + "\n"

        except Exception as e:
            st.warning(f"Could not read {name}: {e}")
            continue 
            
    progress_bar.empty() 
    return all_text

# --- 2. The RAG Engine Builder ---



@st.cache_resource(show_spinner=False)
def configure_rag_engine():
    """
    Fetches data from Drive, embeds it, and stores the Vector Database in RAM.
    """
    with st.spinner("📥 Downloading data and building AI memory..."):
        # 1. Get Secrets
        try:
            folder_id = st.secrets["DRIVE_FOLDER_ID"]
            api_key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            st.error("Missing secrets! Check DRIVE_FOLDER_ID and GOOGLE_API_KEY.")
            return None
            
        # 2. Fetch Text
        raw_text = get_folder_text(folder_id)
        if not raw_text:
            st.error("No files found or folder is empty.")
            return None
            
        # 3. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = text_splitter.split_text(raw_text)
        
        # 4. Create Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        
        # Batching to stay within Rate Limits
        vector_store = None
        batch_size = 5 
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embedding=embeddings)
            else:
                vector_store.add_texts(batch)
            time.sleep(0.5) 
            
        return vector_store

# --- 3. UI and Chat ---

st.markdown('<h1>Minil.Ai</h1>', unsafe_allow_html=True)

li_url = "https://www.linkedin.com/in/jesussantillanminila/"
st.markdown(f"Hi, I am a chatbot built by [Jesus Santillan Minila]({li_url}) to answer questions about his career.")

vector_store_ram = configure_rag_engine()

if vector_store_ram:
    user_question = st.chat_input("Ask me anything about the experience...")

    if user_question: 
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                api_key = st.secrets["GOOGLE_API_KEY"]
                model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
                
                # 1. Define the specific system prompt
                system_prompt = (
                    " You are a professional assistant. Answer the question using the provided context.\n\n"
                    "Guidelines:\n"
                    "1. Be concise but detailed specific.\n"
                    "2. Prioritize hard facts (numbers, skills, dates) over generic descriptions.\n"
                    "3. If the answer is not in the context, redirect the user to a question that you can actually answer.\n\n"
                    "Context:\n{context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # 2. Create the modern 'Stuff' chain
                question_answer_chain = create_stuff_documents_chain(model, prompt)
                
                # 3. Get relevant documents
                docs = vector_store_ram.similarity_search(user_question, k=5)
                
                # 4. Run the chain
                response = question_answer_chain.invoke({"context": docs, "input": user_question})
                
                st.write(response)
else:
    st.info("System initializing or awaiting configuration.")