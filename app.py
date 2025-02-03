import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader  # For loading PDF files from a directory
from langchain_community.vectorstores import FAISS  # For storing and retrieving vector embeddings
from langchain_core.prompts import ChatPromptTemplate  # For creating prompt templates
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For generating embeddings using Google's Generative AI
import google.generativeai as genai  # Google Generative AI SDK for API configuration
from langchain_groq import ChatGroq  # For using Groq model integrations
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting documents into smaller chunks
from langchain.chains.combine_documents import create_stuff_documents_chain  # For combining documents in a chain
from langchain.chains import create_retrieval_chain  # For creating a retrieval-based chain

# Load environment variables from .env file
load_dotenv()

# Load the GROQ and GEMINI API keys from environment variables
groq_api_key = os.environ['GROQ_API_KEY']

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Configure Google Generative AI

# Streamlit app title
st.title("Chatgroq With Llama3 Demo")

# Initialize the ChatGroq model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Define the first prompt template for question answering
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Define a second prompt template with the same meaning, rephrased
prompt_alternative = ChatPromptTemplate.from_template(
    """
    Using the given context, answer the questions as accurately as possible.
    Respond solely based on the provided information.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Function to perform vector embedding and store the embeddings in session state
def vector_embedding():
    if "vectors" not in st.session_state:
        # Initialize embeddings using Google's Generative AI
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load PDF documents from the specified directory
        st.session_state.loader = PyPDFDirectoryLoader("./PDFS")
        st.session_state.docs = st.session_state.loader.load()

        # Split documents into smaller chunks for better embedding
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:20]
        )

        # Create vector embeddings for the documents
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

# Input for user question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to embed documents into vectors
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# If a question is entered, process the query
if prompt1:
    # Create a chain for combining documents with the LLM and prompt
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Retrieve relevant documents using the vector store
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure response time for the query
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.process_time() - start
    st.write(f"Response time: {response_time:.2f} seconds")

    # Display the LLM's response
    st.write(response['answer'])

    # Show relevant documents with a Streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write("--------------------------------")
