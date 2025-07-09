from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm


model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text(text, chunk_size=300):
    paragraphs = text.split("\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + " "
        else:
            chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks

def embed_and_upload_to_pinecone(chunks, index_name):
    pc = Pinecone(api_key="pcsk_57fk27_HeU3LL9h8u5YAkxUG2dDR8X2V71NXJhJc8RzreCAyLY75bzAwcn9NUTcTqjfpKK")

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(index_name)

    embeddings = model.encode(chunks).tolist()
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    meta = [{"text": chunk} for chunk in chunks]

    batch_size = 100
    for i in tqdm(range(0, len(embeddings), batch_size)):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = embeddings[i:i + batch_size]
        batch_meta = meta[i:i + batch_size]

        index.upsert(vectors=zip(batch_ids, batch_vectors, batch_meta))

def retrieve_relevant_chunks(query, index_name="dr-book-index", top_k=3):
    # Connect to Pinecone
    pc = Pinecone(api_key="pcsk_57fk27_HeU3LL9h8u5YAkxUG2dDR8X2V71NXJhJc8RzreCAyLY75bzAwcn9NUTcTqjfpKK")

    # Load index
    index = pc.Index(index_name)

    # Embed the query
    query_vector = model.encode(query).tolist()

    #  Query Pinecone
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    #  Extract text chunks
    contexts = [match["metadata"]["text"] for match in results["matches"]]
    return "\n\n".join(contexts)




