import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="📚 Chat with PDF (Gemini 2.0 Flash)", page_icon="🤖")
st.title("💬 Chat with your PDF using Gemini 2.0 Flash")

# Step 1: Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Step 3: Build FAISS vector store
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("faiss_index")

# Step 4: Ask Gemini 2.0 Flash
def ask_gemini(context, question):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
You are a helpful assistant. Answer the question using the context below.
If the answer is not in the context, say "Sorry, I couldn't find that in the document."

Context:
{context}

Question:
{question}

Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# Step 5: Search FAISS and ask Gemini
def user_input(question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = ask_gemini(context, question)
    st.write("### 🤖 Gemini 2.0 Flash Answer:")
    st.write(answer)

# Streamlit UI
def main():
    with st.sidebar:
        st.header("📂 Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)

        if st.button("🚀 Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("✅ PDFs processed and indexed!")
            else:
                st.warning("Please upload at least one PDF.")

    question = st.text_input("🔎 Ask a question from the uploaded PDF(s):")
    if question:
        user_input(question)

if __name__ == "__main__":
    main()
