import streamlit as st
import os
import tempfile
import json
from langchain_google_community import GoogleDriveLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# --- FIX IS HERE: Updated Import ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page Config
st.set_page_config(page_title="Professional Bot", layout="wide")
st.markdown('<h1>Minil.Ai</h1>', unsafe_allow_html=True)
li_url = "https://www.linkedin.com/in/jesussantillanminila/"
st.markdown(f"Hi, I am a chatbot built by [Jesus Santillan Minila]({li_url}) to answer questions about his career.")

# 1. Setup Credentials
def get_service_account_file():
    if "gcp_service_account" not in st.secrets:
        st.error("Missing 'gcp_service_account' in secrets.")
        st.stop()
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_file:
        json.dump(dict(st.secrets["gcp_service_account"]), temp_file)
        temp_file.flush()
        return temp_file.name

# 2. Load and Process Data
@st.cache_resource(show_spinner=True)
def load_and_process_data():
    try:
        service_account_path = get_service_account_file()
        folder_id = st.secrets["DRIVE_FOLDER_ID"]
        
        st.write("🔄 Loading documents from Google Drive...")
        
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            service_account_key=service_account_path,
            recursive=True
        )
        
        docs = loader.load()
        
        if not docs:
            st.error("No documents found! Did you share the folder with the service account email?")
            st.stop()
            
        # Split Text using the imported splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("Missing GOOGLE_API_KEY in secrets.")
            st.stop()
            
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        st.success(f"✅ Loaded {len(docs)} documents successfully!")
        
        os.unlink(service_account_path)
        return vectorstore

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Initialize Chat Chain
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_and_process_data()

# 3. Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0,
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)

# 4. Create RAG Chain
retriever = st.session_state.vectorstore.as_retriever()

system_prompt = (
                    "You are a professional assistant. Answer the question using the provided context.\n\n"
                    "Guidelines:\n"
                    "1. Be concise but detailed specific.\n"
                    "2. Prioritize hard facts (numbers, skills, dates) over generic descriptions.\n"
                    "3. If the answer is not in the context, redirect the user to a question that you can actually answer.\n\n"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("Ask a question about your docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt_text})
            answer = response["answer"]
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})