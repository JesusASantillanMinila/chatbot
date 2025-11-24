import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import time

# --- Page Config ---
st.set_page_config(page_title="Resume Bot", layout="wide")

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
                    all_text += page.extract_text() + "\n"

            elif mime == 'application/vnd.google-apps.document':
                request = service.files().export_media(fileId=file_id, mimeType='text/plain')
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    _, done = downloader.next_chunk()
                all_text += fh.getvalue().decode('utf-8') + "\n"

        except Exception:
            continue # Skip files that fail to read
            
    progress_bar.empty() # Hide bar when done
    return all_text

# --- 2. The "Brain" Builder (Cached) ---

@st.cache_resource(show_spinner=False)
def configure_rag_engine():
    """
    This function runs ONCE when the app starts.
    It fetches data from Drive, embeds it, and stores the Vector Database in RAM.
    """
    with st.spinner("📥 Downloading resume data from Drive and building AI memory..."):
        
        # 1. Get the Folder ID from Secrets
        try:
            folder_id = st.secrets["DRIVE_FOLDER_ID"]
        except FileNotFoundError:
            st.error("Missing secrets! Please add DRIVE_FOLDER_ID to .streamlit/secrets.toml")
            return None
            
        # 2. Fetch Text
        raw_text = get_folder_text(folder_id)
        if not raw_text:
            st.error("No files found in the Drive folder (or folder is empty).")
            return None
            
        # 3. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        
        # 4. Create Vector Store (Embeddings)
        #    Using 'text-embedding-004' to avoid legacy model errors
        api_key = st.secrets["GOOGLE_API_KEY"]
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        
        # Process in small batches to be kind to the API limits
        vector_store = None
        batch_size = 10 
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embedding=embeddings)
            else:
                vector_store.add_texts(batch)
            time.sleep(1) # Brief pause to respect rate limits
            
        return vector_store

# --- 3. Main Chat Interface ---

st.markdown('<h1>Minil.Ai</h1>', unsafe_allow_html=True)

st.markdown("""
Hi, I am a chatbot built by **[Jesus Santillan Minila]**(https://www.linkedin.com/in/jesussantillanminila/) to answer questions about his  
 professional background, technical skills, or project experience.
""")

# Initialize the brain automatically
vector_store_ram = configure_rag_engine()

if vector_store_ram:
    # Chat input
    user_question = st.chat_input("Ask me anything about the resume...")

    if user_question:
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                api_key = st.secrets["GOOGLE_API_KEY"]
                # Using 'gemini-2.5-flash' as the standard model
                model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)
                
                prompt_template = """
                
                You are a professional assistant. Answer the question using the provided context.
                
                Guidelines:
                1. Be concise but detailed specific.
                2. Prioritize hard facts (numbers, skills, dates) over generic descriptions.
                3. If the answer is not in the context, say "I'm sorry, my responses are limited. You must ask the right questions."
                
                Context:
                {context}
                
                Question: 
                {question}
                
                Answer:
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
                
                docs = vector_store_ram.similarity_search(user_question)
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                
                st.write(response["output_text"])

else:
    st.info("Please ensure your secrets are configured and your Drive folder has files.")