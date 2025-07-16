from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Load FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vector_db", embedding_model, allow_dangerous_deserialization=True)

def load_system_prompt():
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()
    
system_prompt = load_system_prompt()

# Load Groq LLM
llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-2.5-flash"
)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt + "\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)


# Create chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt_template}
)

def send_to_gimini_with_rag(question):
    try:
        print(f"Processing question: {question}")  # Debug
        response = qa_chain({"query": question})  # Changed 'question' to 'query'
        print(f"Response: {response}")  # Debug
        answer = response['result']
        return answer
    except Exception as e:
        print("Error:", e)
        return f"Error: {str(e)}"




    