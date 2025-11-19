import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import time  # <--- Added this library to handle the pauses

# --- Page Config ---
st.set_page_config(page_title="Resume Chatbot", layout="wide")

# --- Header ---
st.markdown("""
<style>
.big-font { font-size:30px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">🤖 Chat with my Resume</p>', unsafe_allow_html=True)

# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the text into manageable chunks."""
    # Increased chunk size slightly to reduce total number of API calls needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    UPDATED: Uses the newer 'text-embedding-004' model which has a valid free tier.
    """
    api_key = st.secrets["GOOGLE_API_KEY"]
    # CHANGED: from "models/embedding-001" to "models/text-embedding-004"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    
    vector_store = None
    batch_size = 5  
    
    progress_text = "Embedding resume sections... please wait."
    my_bar = st.progress(0, text=progress_text)
    total_chunks = len(text_chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch = text_chunks[i:i + batch_size]
        
        if vector_store is None:
            vector_store = FAISS.from_texts(batch, embedding=embeddings)
        else:
            vector_store.add_texts(batch)
            
        progress = min((i + batch_size) / total_chunks, 1.0)
        my_bar.progress(progress, text=f"Processing batch {i//batch_size + 1}...")
        
        # Keep the sleep to be safe, though the new model is often more generous
        time.sleep(1) 
    
    vector_store.save_local("faiss_index")
    my_bar.empty()

def get_conversational_chain():
    """
    Creates the chain that connects the LLM (Gemini) to the prompt.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say, "The resume does not provide this information," don't make up the wrong answer.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    api_key = st.secrets["GOOGLE_API_KEY"]
    # standard gemini-pro model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """
    Handles the user query.
    """
    api_key = st.secrets["GOOGLE_API_KEY"]
    # CHANGED: from "models/embedding-001" to "models/text-embedding-004"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("reply: ", response["output_text"])
    except RuntimeError as e:
         st.error("Error: The index is missing. Please upload and process the resume first.")

# --- Sidebar (Resume Upload) ---
with st.sidebar:
    st.title("Upload Resume")
    pdf_docs = st.file_uploader("Upload your PDF Resume here", accept_multiple_files=True)
    if st.button("Process Resume"):
        if not pdf_docs:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

# --- Main Chat Interface ---
user_question = st.text_input("Ask a question about the candidate (e.g., 'What are their Python skills?')")

if user_question:
    user_input(user_question)