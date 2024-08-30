import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv
from pypdf.errors import PdfReadError

load_dotenv()

# Load the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Setting up the environment..."):
            try:
                st.session_state.embeddings = NVIDIAEmbeddings()
                st.session_state.loader = PyPDFDirectoryLoader("trip")  # Data Ingestion
                docs = []
                for file_path in st.session_state.loader.file_paths:
                    try:
                        with open(file_path, "rb") as file:
                            doc = st.session_state.loader.parser.parse(file)
                            docs.extend(doc)
                    except (PdfReadError, PdfStreamError) as e:
                        st.warning(f"Warning: Could not read file {file_path}. Skipping. Error: {e}")
                
                st.session_state.docs = docs
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting

                if not st.session_state.final_documents:
                    st.error("Error: No documents were loaded or split. Please check the document directory.")
                    return

                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
            except Exception as e:
                st.error(f"An error occurred during vector embedding: {e}")
                st.stop()

# Streamlit page setup
st.set_page_config(page_title="Uttarakhand Traveling", layout="wide", page_icon="🏔️")
st.title("TRIP PLANNING FOR UTTARAKHAND")

# Load language model
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Define the prompt
prompt = ChatPromptTemplate.from_template(
    """
    Provide a detailed trip plan for the given destination and number of days.
    Include information about local cuisine, hotels, and links to important places.
    End with a cheerful journey message.
    <context>
    {context}
    </context>
    Destination: {input}
    Days: {days}
    """
)

# Initialize vector embeddings
vector_embedding()

# Main application interface
st.write("Welcome to the Uttarakhand Trip Planner")
st.image("utta.jpg", caption="Sunrise by the mountains")
with st.sidebar:
    st.subheader("Plan Your Trip", divider="rainbow")
    prompt1 = st.text_input("Enter your Destination", help="Example: Nainital, Mussoorie")
    prompt2 = st.text_input("Enter Number of Days", help="Example: 5, 7")
    maps = f"https://www.google.com/maps/place/{prompt1}"

if not prompt1 or not prompt2:
    st.error("Please fill out both fields.")
else:
    with st.spinner("Processing your request..."):
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1, 'days': prompt2})
            processing_time = time.process_time() - start
            
            st.success(f"Response received in {processing_time:.2f} seconds")
            st.write(response['answer'])
            st.button("Click for maps", on_click=lambda: st.markdown(maps, unsafe_allow_html=True))
        
        except Exception as e:
            st.error(f"An error occurred during request processing: {e}")

if st.button("Clear"):
    st.session_state.clear()
