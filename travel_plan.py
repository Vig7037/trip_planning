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
load_dotenv()

# Load the Groq API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Setting up the environment..."):
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("trip")  # Data Ingestion
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
st.set_page_config(page_title="Uttarakhand Traveling", layout="wide", page_icon="🏔️")
st.title("TRIP PLANNING FOR UTTARAKHAND")

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

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
maps="https://www.google.com/maps/place/"
vector_embedding()

st.write("Welcome to the Uttarakhand Trip Planner")
st.image("utta.jpg", caption="Sunrise by the mountains")
with st.sidebar:
    st.subheader("Plan Your Trip",divider="rainbow")
    prompt1 = st.text_input("Enter your Destination", help="Example: Nainital, Mussoorie")
    prompt2 = st.text_input("Enter Number of Days", help="Example: 5, 7")
    maps=maps+prompt1

if not prompt1 or not prompt2:
    st.error("Please fill out both fields.")
else:
    with st.spinner("Processing your request..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1, 'days': prompt2})
        processing_time = time.process_time() - start
        st.success(f"Response received in {processing_time:.2f} seconds")

        st.write(response['answer'])
        st.link_button("Click for maps", maps)

if st.button("Clear"):
    st.session_state.clear()
