from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data):
    """Loads PDF files from a given directory."""
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Step 2: Split text into chunks
def create_chunks(extracted_data):
    """Splits extracted documents into smaller text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks  # ✅ Added return statement

# Step 3: Load embedding model
def get_embedding_model():
    """Loads Hugging Face sentence transformer embeddings."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# Step 4: Create FAISS vectorstore
def create_vectorstore(documents, embedding_model, db_path):
    """Creates and saves FAISS vectorstore from document embeddings."""
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(db_path)
    return db

# Main execution
def main():
    # Load and process documents
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)

    # Load embedding model
    embedding_model = get_embedding_model()

    # Define FAISS database path
    DB_FAISS_PATH = "vectorstore/db_faiss"

    # Create and save FAISS vectorstore
    db = create_vectorstore(text_chunks, embedding_model, DB_FAISS_PATH)

if __name__ == "__main__":
    main()
