# 🧠 Diabetic Retinopathy Detection Medical Chatbot 

This part of the project acts as a **clinical assistant** for healthcare professionals, offering evidence-based information and treatment guidance related to **diabetic retinopathy (DR)**.

## 🔍 Project Features

- ✅ RAG (Retrieval-Augmented Generation) powered medical assistant  
- ✅ Answers questions using a custom-built knowledge base from DR literature  
- ✅ Streamlit frontend for interactive use  
- ✅ Gemini 2.5 Flash LLM for high-quality medical answers  
- ✅ Built with LangChain + HuggingFace + FAISS

## 🧱 Tech Stack

| Component                   | Description                                             |
|----------------------------|---------------------------------------------------------|
| **LangChain**              | Framework for chaining LLMs with vector retrieval       |
| **Gemini 2.5 Flash**       | Google LLM via LangChain for medical reasoning          |
| **MiniLM-L12-v2**          | HuggingFace embedding model for semantic similarity     |
| **FAISS**                  | Local vector database for fast retrieval                |
| **Streamlit**              | Web interface for chatting with the model               |
| **PDF + Data Injection**   | Custom section chunking and extraction                  |
 
---

## File & Folder Descriptions

├── data/                    # Source PDFs
├── vector_db/              # FAISS index
├── build_knowledge.py      # Build knowledge base
├── chatbot.py              # Backend LLM + RAG logic
├── app.py                  # Streamlit UI
├── evaluation.py           # Model testing/evaluation
├── system_prompt.txt       # Gemini prompt instructions
├── .env                    # API keys
├── requirements.txt
├── README.md

## Demo

Below are reference images showing the chatbot in action:

![Chatbot Demo 1](images/1.png)
![Chatbot Demo 2](images/2.png)


## Getting Started

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/Diabetic-Retinopathy-Detection.git
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Build knowledge base:**
   ```
   python build_knowledge.py
   ```
4. **Run the application:**
   ```
   streamlit run app.py  
   ```
5. **Access the web interface:**
   Open your browser and go to `http://localhost:8501`

