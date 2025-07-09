
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from rag_engine import extract_text_from_pdf, split_text, embed_and_upload_to_pinecone

pdf_path = os.path.join(os.path.dirname(__file__), "978-981-10-3509-8.pdf")

index_name = "dr-book-index"

print("[1] Extracting text from PDF...")
text = extract_text_from_pdf(pdf_path)

print("[2] Splitting text into chunks...")
chunks = split_text(text, chunk_size=300)

print("[3] Uploading to Pinecone...")
embed_and_upload_to_pinecone(chunks, index_name)

print("Knowledge base built successfully!")
