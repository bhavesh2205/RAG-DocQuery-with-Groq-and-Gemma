import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY")

# app configuration
st.set_page_config(page_title="RAG-QA with Groq and Gemma", layout="wide")

# sidebar
st.sidebar.title("Configuration")
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
)

# app title
st.title("RAG-QA with Groq and Gemma")

# upload pdfs
uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

# process documents
if uploaded_files:
    try:
        documents = []
        for uploaded_file in uploaded_files:
            # save uploaded file to a temporary file and load it
            with open(uploaded_file.name, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(temp_file.name)
                documents.extend(loader.load())
    
        # split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Error processing documents: {e}")

# create vector db
if 'docs' in locals() and embedding_model:
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vector_store = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Error creating vector db: {e}")

## provide a way to ask questions
st.markdown("#### Ask Questions")
query = st.text_input("Enter your query:")

if st.button("Get Response") and query and 'vector_store' in locals():
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        # define the prompt template
        prompt_template = (
            "You are a helpful assistant. Use the following context to answer the question.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        # initialize the ChatGroq model
        model = ChatGroq(model_name="gemma2-9b-it")

        # create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])},
            return_source_documents=True
        )
        
        # response
        response = qa_chain({"query": query})
        st.markdown("#### Answer:")
        st.write(response['result'])

        # relevent sources
        st.subheader("## Source Chunks:")
        for source in response["source_documents"]:
            st.write(source.page_content)
    except Exception as e:
        st.error(f"Error generating response: {e}")

# footer
st.markdown("---")
st.markdown("Powered by Streamlit, Groq, HuggingFace, and LangChain")