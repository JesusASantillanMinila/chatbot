import streamlit as st
import os
import tempfile
from google.oauth2 import service_account
from googleapiclient.discovery import build
import google.generativeai as genai

# --- CONFIGURATION ---
# Access secrets from Streamlit
try:
    DRIVE_FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SERVICE_ACCOUNT_INFO = st.secrets["gcp_service_account"]
except Exception as e:
    st.error(f"Missing secrets: {e}. Please configure .streamlit/secrets.toml")
    st.stop()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# --- GOOGLE DRIVE FUNCTIONS ---
def get_drive_service():
    """Authenticates and returns the Drive API service."""
    creds = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=creds)

def download_docs_from_folder(folder_id):
    """
    Exports Google Docs from a specific folder to local temporary text files.
    Returns a list of file paths.
    """
    service = get_drive_service()
    results = []
    
    # List files in the folder (Query for Google Docs only to keep it simple)
    query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.document' and trashed=false"
    items = service.files().list(q=query, fields="files(id, name)").execute().get('files', [])

    if not items:
        return []

    # Create a temporary directory to store downloaded files
    temp_dir = tempfile.mkdtemp()
    
    status_bar = st.status("Syncing Knowledge Base...", expanded=True)
    
    for file in items:
        file_id = file['id']
        file_name = file['name']
        # status_bar.write(f"Downloading: {file_name}")
        
        # Export Google Doc as Plain Text
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
        response = request.execute()
        
        # Save to temp file
        safe_name = "".join([c for c in file_name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        file_path = os.path.join(temp_dir, f"{safe_name}.txt")
        
        with open(file_path, "wb") as f:
            f.write(response)
        results.append(file_path)
    
    status_bar.update(label="Sync Complete!", state="complete", expanded=False)
    return results

# --- GEMINI RAG FUNCTIONS ---
@st.cache_resource(show_spinner=False)
def initialize_knowledge_base(folder_id):
    """
    Downloads docs, uploads them to Gemini File API, and sets up the model.
    Cached so it doesn't re-run on every interaction.
    """
    # 1. Download Files from Drive
    file_paths = download_docs_from_folder(folder_id)
    
    if not file_paths:
        return None

    # 2. Upload to Gemini File API
    uploaded_files = []
    progress_text = "Indexing documents in Gemini..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, path in enumerate(file_paths):
        # Upload file to Gemini
        gemini_file = genai.upload_file(path=path)
        uploaded_files.append(gemini_file)
        my_bar.progress((i + 1) / len(file_paths), text=progress_text)
    
    my_bar.empty()
    
    # 3. Initialize Model with these files (Managed RAG)
    # Using gemini-2.5-flash for speed and free tier efficiency
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="You are a helpful assistant. Answer questions using the provided context files. If the answer is not in the context, redirect the user to a question that you can actually answer.",
        tools=[{"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "unspecified"}}}]  # Fallback
    )
    
    return uploaded_files

# --- STREAMLIT APP ---
st.markdown('<h1>Minil.Ai</h1>', unsafe_allow_html=True)
li_url = "https://www.linkedin.com/in/jesussantillanminila/"
st.markdown(f"Hi, I am a chatbot built by [Jesus Santillan Minila]({li_url}) to answer questions about his career.")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
    files = initialize_knowledge_base(DRIVE_FOLDER_ID)
    if files:
        # Create a chat session with the uploaded files as history/context
        # Note: Gemini 2.5 allows passing files directly in the history or generation request
        model = genai.GenerativeModel("gemini-2.5-flash")
        st.session_state.chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": files + ["Use these files as your knowledge base."]
                },
                {
                    "role": "model",
                    "parts": ["Understood. I am ready to answer your questions."]
                }
            ]
        )
    else:
        st.error("I could not find any reliable information.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if "chat_session" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_session.send_message(prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"An error occurred: {e}")