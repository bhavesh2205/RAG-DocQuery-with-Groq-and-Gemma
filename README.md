# RAG-QA-with-Groq-and-Gemma

This project is a **Retrieval-Augmented Generation (RAG)** application that leverages Streamlit for an interactive user interface and integrates modern GenAI tools to provide context-aware answers based on provided document content.

## Key Components

### 1. **Document Processing**

- **File Upload**: Users can upload PDF files containing any information.
- **PyPDFLoader**: Extracts text from uploaded PDFs.
- **RecursiveCharacterTextSplitter**: Splits the text into manageable chunks for efficient processing.

### 2. **Vectorization and Storage**

- **HuggingFaceEmbeddings**: Converts text into numerical vectors.
- **FAISS Vector Store**: Stores and retrieves document embeddings for efficient similarity searches.

### 3. **Question Answering**

- **ChatGroq Model**: Utilizes Groq's `gemma-13b` model for generating accurate and context-relevant answers.
- **RetrievalQA Chain**: Combines document retrieval with answer generation.

### 4. **User Interface**

- **Streamlit**: Provides a clean and interactive interface for file uploads, query input, and displaying results.

## Usage

1. Upload PDFs through the Streamlit app interface.
2. Enter a query to ask a question based on the document content.
3. View the generated answer and source references.

## Libraries and Tools

- **Streamlit**: UI development.
- **LangChain**: Document loading, text splitting, and QA chain.
- **HuggingFace Transformers**: Text embeddings.
- **FAISS**: Vector storage and retrieval.
- **Groq Chat Model**: LLM model for answering queries.

## How It Works

1. **Document Ingestion**: Upload PDFs which are processed into chunks.
2. **Embedding Creation**: Text chunks are converted into embeddings using HuggingFace models.
3. **Query Processing**: User queries are matched against document embeddings to retrieve relevant content.
4. **Answer Generation**: The retrieved context is passed to the Groq model to generate an answer.

## Prerequisites

- Python 3.9+
- Required Python libraries: `streamlit`, `langchain`, `faiss-cpu`, `transformers`, `groq`.

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```


