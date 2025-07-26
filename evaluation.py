import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score
from chatbot import send_to_gimini_with_rag  

embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
sbert = SentenceTransformer(embedding_model_name)

db = FAISS.load_local("vector_db", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Sample questions and ground-truth answers 
eval_set = [

    {"question": "How is the classification of NPDR determined?",
     "expected_keywords": ["microaneurysms", "hemorrhages", "irma", "venous caliber"]}
     ,
    {"question": "What is the role of the neurovascular unit in diabetic retinopathy?",
     "expected_keywords": ["neurons", "müller cells", "glia", "endothelial", "neurovascular"]}
     ,
    {"question": "What is metabolic memory in DR and its biological basis?",
     "expected_keywords": ["glycemic memory", "epigenetic", "AGEs", "oxidative damage"]},
    {"question": "When is panretinal photocoagulation preferred over anti-VEGF?",
     "expected_keywords": ["prp", "compliance", "follow-up", "laser", "monthly injection"]},
    {"question": "What are some VEGF-independent therapies being explored for DR?",
     "expected_keywords": ["tnf-alpha", "kallikrein", "lp-pla2", "bdnf", "ngf"]},
    {
        "question": "What is diabetic retinopathy current treatment?",
        "expected_keywords": ["anti-VEGF", "laser photocoagulation", "steroids", "PRP", "injections"]
    }
]

def compute_faithfulness(answer, retrieved_docs):
    context = " ".join([doc.page_content for doc in retrieved_docs])
    emb_answer = sbert.encode(answer, convert_to_tensor=True)
    emb_context = sbert.encode(context, convert_to_tensor=True)
    similarity = util.cos_sim(emb_answer, emb_context).item()
    return similarity

def compute_precision_at_k(retrieved_docs, expected_keywords):
    relevant_docs = 0
    for doc in retrieved_docs:
        if any(keyword.lower() in doc.page_content.lower() for keyword in expected_keywords):
            relevant_docs += 1
    return relevant_docs / len(retrieved_docs)

total_precision, total_faithfulness = [], []


for i, sample in enumerate(eval_set):
    question = sample["question"]
    expected_keywords = sample["expected_keywords"]

    print(f"\nQuestion {i+1}: {question}")
    
    answer = send_to_gimini_with_rag(question)
    
    retrieved = retriever.get_relevant_documents(question)

    sim_score = compute_faithfulness(answer, retrieved)
    print(f"Faithfulness Score (cosine sim): {sim_score:.3f}")

    prec = compute_precision_at_k(retrieved, expected_keywords)
    print(f"Precision@{len(retrieved)}: {prec:.2f}")

    total_faithfulness.append(sim_score)
    total_precision.append(prec)

print("\n--- Evaluation Summary ---")
print(f"Avg Faithfulness Score: {np.mean(total_faithfulness):.3f}")
print(f"Avg Precision@{len(retrieved)}: {np.mean(total_precision):.2f}")
