from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv() 

# 1. Load PDF
def load_documents():
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

# 2. Clean Text
def clean_text(text):
    return text.strip().replace("\n", " ")

# 3. Split into Chunks
def split_documents(docs):
    for doc in docs:
        doc.page_content = clean_text(doc.page_content) 

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

# 4. Convert to Embeddings + Store in FAISS
def store_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vector_db")

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")

    print("Splitting documents into chunks...")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    print("Storing chunks in vector database...")
    store_vector_db(chunks)
    print("Vector database stored successfully.")
