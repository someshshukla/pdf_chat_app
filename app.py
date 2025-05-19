import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import shutil

# --- CONFIGURATION ---
CHAT_MODEL_NAME = "models/gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "models/embedding-001"
FAISS_INDEX_ROOT_DIR = "faiss_indexes"
# --- END CONFIGURATION ---

st.set_page_config(page_title="Multi-DocQuery with Gemini", layout="wide")

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)

if not os.path.exists(FAISS_INDEX_ROOT_DIR):
    os.makedirs(FAISS_INDEX_ROOT_DIR)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
api_key_configured = False
embeddings_object_global = None

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        embeddings_object_global = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, task_type="retrieval_document")
        api_key_configured = True
    except Exception as e:
        st.error(f"Error configuring Google API or Embeddings: {e}")
        st.sidebar.error("Error configuring Google API.")
else:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.sidebar.error("Google API Key not found in .env file.")

if api_key_configured:
    st.sidebar.success(f"API Key loaded. Chat: {CHAT_MODEL_NAME.split('/')[-1]}")
elif not GOOGLE_API_KEY:
    st.stop()

def get_pdf_text(pdf_file_obj):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file_obj)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading PDF content for {pdf_file_obj.name if hasattr(pdf_file_obj, 'name') else 'uploaded file'}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_and_save_vector_store(text_chunks, pdf_name, embeddings_obj):
    if not text_chunks:
        st.warning(f"No text chunks for {pdf_name}, cannot create vector store.")
        return None
    if not embeddings_obj:
        st.error("Embeddings object not available. Cannot create vector store.")
        return None
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings_obj)
        sanitized_name = sanitize_filename(pdf_name)
        save_path = os.path.join(FAISS_INDEX_ROOT_DIR, f"{sanitized_name}_index")
        vector_store.save_local(save_path)
        st.success(f"Vector store created and saved for {pdf_name}")
        return vector_store
    except Exception as e:
        st.error(f"Error creating/saving vector store for {pdf_name}: {e}")
    return None

def load_vector_store_from_disk(pdf_name, embeddings_obj):
    if not embeddings_obj:
        st.error("Embeddings object not available. Cannot load vector store.")
        return None
    sanitized_name = sanitize_filename(pdf_name)
    load_path = os.path.join(FAISS_INDEX_ROOT_DIR, f"{sanitized_name}_index")
    if os.path.exists(load_path):
        try:
            
            return FAISS.load_local(load_path, embeddings_obj, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Could not load FAISS index from {load_path} for {pdf_name}. Will reprocess if uploaded. Error: {e}")
    return None

def get_conversational_chain():
    if not api_key_configured:
        return None
    prompt_template_str = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say, "The answer is not available in the provided document."
    Do not provide a wrong answer. Context:\n {context}\nQuestion: \n{question}\nAnswer:"""
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME, temperature=0.3, convert_system_message_to_human=True)
    prompt = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "processed_pdf_names" not in st.session_state:
    st.session_state.processed_pdf_names = []
if "active_pdf_name" not in st.session_state:
    st.session_state.active_pdf_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
    if api_key_configured:
        st.session_state.chain = get_conversational_chain()
elif api_key_configured and not st.session_state.chain:
    st.session_state.chain = get_conversational_chain()


st.header(f"📄 Multi-DocQuery with {CHAT_MODEL_NAME.split('/')[-1]}")

with st.sidebar:
    st.title("Document Management")
   

    if api_key_configured and embeddings_object_global:
        st.subheader("Upload New PDF")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key=f"pdf_uploader_{len(st.session_state.processed_pdf_names)}")

        if uploaded_file:
            pdf_name = uploaded_file.name
            if pdf_name not in st.session_state.vector_stores:
                with st.spinner(f"Checking/Processing '{pdf_name}'..."):
                    vs_from_disk = load_vector_store_from_disk(pdf_name, embeddings_object_global)
                    if vs_from_disk:
                        st.session_state.vector_stores[pdf_name] = vs_from_disk
                        if pdf_name not in st.session_state.processed_pdf_names:
                             st.session_state.processed_pdf_names.append(pdf_name)
                        st.success(f"Loaded existing vector store for {pdf_name} from disk.")
                    else:
                        raw_text = get_pdf_text(uploaded_file)
                        if raw_text and raw_text.strip():
                            text_chunks = get_text_chunks(raw_text)
                            if text_chunks:
                                vector_store = create_and_save_vector_store(text_chunks, pdf_name, embeddings_object_global)
                                if vector_store:
                                    st.session_state.vector_stores[pdf_name] = vector_store
                                    if pdf_name not in st.session_state.processed_pdf_names:
                                        st.session_state.processed_pdf_names.append(pdf_name)
                                else:
                                    st.error(f"Failed to create vector store for {pdf_name}.")
                            else:
                                st.warning(f"No text chunks generated from {pdf_name}.")
                        else:
                            st.error(f"Could not extract text from {pdf_name}.")
            
            if pdf_name in st.session_state.vector_stores and st.session_state.active_pdf_name != pdf_name:
                st.session_state.active_pdf_name = pdf_name
                st.session_state.messages = []
                st.success(f"Switched active document to: {pdf_name}")
                

        st.divider()
        st.subheader("Select Active Document")
        
        available_options = sorted(list(st.session_state.vector_stores.keys()))

        if not available_options:
            st.caption("No PDFs processed. Upload a PDF.")
        else:
            current_active_index = 0
            if st.session_state.active_pdf_name and st.session_state.active_pdf_name in available_options:
                current_active_index = available_options.index(st.session_state.active_pdf_name)
            elif available_options and not st.session_state.active_pdf_name: 
                st.session_state.active_pdf_name = available_options[0]
                current_active_index = 0
                st.session_state.messages = [] 

            selected_pdf_name_from_dropdown = st.selectbox(
                "Chat with:",
                options=available_options,
                index=current_active_index,
                key="active_pdf_dropdown" 
            )

            if selected_pdf_name_from_dropdown and selected_pdf_name_from_dropdown != st.session_state.active_pdf_name:
                st.session_state.active_pdf_name = selected_pdf_name_from_dropdown
                st.session_state.messages = [] 
                st.info(f"Active document switched to: {st.session_state.active_pdf_name}")
                st.rerun() 
            elif selected_pdf_name_from_dropdown and selected_pdf_name_from_dropdown not in st.session_state.vector_stores:
                
                 with st.spinner(f"Loading missing vector store for {selected_pdf_name_from_dropdown}..."):
                    vs = load_vector_store_from_disk(selected_pdf_name_from_dropdown, embeddings_object_global)
                    if vs:
                        st.session_state.vector_stores[selected_pdf_name_from_dropdown] = vs
                        st.success(f"Successfully loaded vector store for {selected_pdf_name_from_dropdown}.")
                       
                    else:
                        st.error(f"Could not load vector store for {selected_pdf_name_from_dropdown}. Please re-upload.")
                        
                        if selected_pdf_name_from_dropdown in st.session_state.vector_stores:
                            del st.session_state.vector_stores[selected_pdf_name_from_dropdown]
                        st.session_state.active_pdf_name = None 
                        st.rerun()

    else:
        st.sidebar.warning("PDF management disabled until API key and embeddings are configured.")


# --- MAIN CHAT INTERFACE ---
if st.session_state.active_pdf_name and st.session_state.active_pdf_name in st.session_state.vector_stores:
    active_vs = st.session_state.vector_stores.get(st.session_state.active_pdf_name)
    st.subheader(f"Chatting with: {st.session_state.active_pdf_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask about {st.session_state.active_pdf_name}..."):
        if not active_vs: 
            st.warning("Active PDF's vector store not found. Please select or re-upload.")
        elif not st.session_state.chain:
            st.warning("Conversational chain is not ready. Check API/model config.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                try:
                   
                    docs = active_vs.similarity_search(prompt, k=3) # Assumes FAISS handles embedding the query
                    
                    chain = st.session_state.chain
                    response_dict = chain.invoke({"input_documents": docs, "question": prompt})
                    answer = response_dict.get("output_text", "Error: Could not find 'output_text' in response.")
                except Exception as e:
                    answer = f"Sorry, an error occurred: {e}"
                    st.error(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
elif st.session_state.active_pdf_name and st.session_state.active_pdf_name not in st.session_state.vector_stores:
    st.warning(f"Vector store for '{st.session_state.active_pdf_name}' is not loaded. Please re-select or re-upload.")
else:
    st.info("Please upload a PDF and/or select an active document from the sidebar to begin chatting.")