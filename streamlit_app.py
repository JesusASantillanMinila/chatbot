import streamlit as st
import google.generativeai as genai
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import io

# --- CONFIGURATION ---
# Configure Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- CLASS DEFINITIONS ---

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for ChromaDB using Google's Gemini Embedding API.
    Avoids LangChain dependencies.
    """
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/text-embedding-004" # Or "models/embedding-001"
        
        # Gemini API expects a specific format. We batch requests for efficiency.
        # Note: Production apps might need more robust batching logic (e.g. max 100 at a time).
        embeddings = []
        for text in input:
            response = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response['embedding'])
        return embeddings

def get_drive_service():
    """Authenticates with Google Drive using Service Account from Streamlit Secrets."""
    creds_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=creds)

def load_documents_from_drive(folder_id):
    """
    Fetches all Google Docs from a specific folder and exports them as text.
    Returns a list of dictionaries: [{'text': ..., 'source': ...}]
    """
    service = get_drive_service()
    results = []
    
    # List files in the folder
    query = f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.document' and trashed = false"
    # Fields parameter limits response size
    response = service.files().list(q=query, fields="files(id, name)").execute()
    files = response.get('files', [])

    if not files:
        st.warning("No Google Docs found in the specified folder.")
        return []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(files):
        file_id = file['id']
        file_name = file['name']
        status_text.text(f"Processing: {file_name}")

        try:
            # Export Google Doc as plain text
            request = service.files().export_media(fileId=file_id, mimeType='text/plain')
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            text_content = fh.getvalue().decode('utf-8')
            
            # Simple chunking (Split by paragraphs or character limit)
            # For deeper RAG, use a recursive character splitter function.
            # Here we do a simple split to keep it dependency-free.
            chunk_size = 1000 
            chunks = [text_content[j:j+chunk_size] for j in range(0, len(text_content), chunk_size)]
            
            for chunk in chunks:
                if chunk.strip(): # Ignore empty chunks
                    results.append({"text": chunk, "source": file_name})
                    
        except Exception as e:
            st.error(f"Error processing {file_name}: {e}")
        
        progress_bar.progress((i + 1) / len(files))

    status_text.text("Ingestion Complete!")
    progress_bar.empty()
    return results

def initialize_vector_db(documents):
    """
    Initializes ChromaDB, creates embeddings via Gemini, and stores them.
    """
    client = chromadb.Client() # Ephemeral (in-memory) for this session. Use PersistentClient for disk storage.
    
    # Create a collection with our custom Gemini embedding function
    collection = client.create_collection(
        name="drive_docs",
        embedding_function=GeminiEmbeddingFunction()
    )

    ids = [str(i) for i in range(len(documents))]
    texts = [doc['text'] for doc in documents]
    metadatas = [{"source": doc['source']} for doc in documents]

    # Add data to Chroma
    # Note: Chroma handles the embedding calls automatically via the function we passed
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    return collection

def query_llm(query, retrieved_context):
    """
    Sends the user query + retrieved context to Gemini Pro for an answer.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-001') 
    
    prompt = f"""
    You are a helpful assistant. Answer the question based ONLY on the provided context.
    
    Context:
    {retrieved_context}
    
    Question: 
    {query}
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- MAIN APP UI ---

st.title("🤖 Google Drive RAG Bot")
st.caption("Ask questions about your Google Docs without LangChain")

# 1. Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Initialize Vector DB (Run once)
if "vector_db" not in st.session_state:
    with st.spinner("Connecting to Google Drive and ingesting documents..."):
        folder_id = st.secrets["DRIVE_FOLDER_ID"]
        docs = load_documents_from_drive(folder_id)
        if docs:
            st.session_state.vector_db = initialize_vector_db(docs)
            st.success(f"Ready! Loaded {len(docs)} text chunks.")
        else:
            st.error("Could not load documents. Check your Folder ID and permissions.")

# 3. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input & Processing
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Pipeline
    if "vector_db" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # A. Retrieve relevant docs
                collection = st.session_state.vector_db
                results = collection.query(
                    query_texts=[prompt],
                    n_results=3 # Number of chunks to retrieve
                )
                
                # B. Format context
                context_text = "\n\n".join(results['documents'][0])
                sources = set([m['source'] for m in results['metadatas'][0]])
                
                # C. Generate Answer
                answer = query_llm(prompt, context_text)
                
                # Display answer + sources
                st.markdown(answer)
                st.markdown(f"**Sources:** *{', '.join(sources)}*")
                
                # Append to history
                st.session_state.messages.append({"role": "assistant", "content": answer})