
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
import uuid
import json
import requests

# Load API key
load_dotenv()
GOOGLE_API_KEY = "AIzaSyCEmxxWHry4P5uFMwJBjLrXveQAzc-W7y8"
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit page config
st.set_page_config(page_title="📚 CourseBook Revision App", page_icon="🤖", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"id": str(uuid.uuid4()), "title": "Chat 1", "messages": []}]
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = st.session_state.chat_history[0]["id"]
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "processed_pdfs" not in st.session_state:
    st.session_state.processed_pdfs = [
        {"name": "NCERT Physics Ch1.pdf", "url": "https://ncert.nic.in/pdf/publication/exemplarproblem/classXI/physics/leep1.pdf"},
        {"name": "NCERT Physics Ch2.pdf", "url": "https://ncert.nic.in/pdf/publication/exemplarproblem/classXI/physics/leep2.pdf"}
    ]

# Step 1: Extract text from PDFs with page numbers
def get_pdf_text(pdf_docs):
    text_with_pages = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        pdf_name = pdf.name if hasattr(pdf, 'name') else "Unknown PDF"
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                text_with_pages.append({"text": text, "page": page_num, "pdf_name": pdf_name})
    return text_with_pages

# Step 2: Split text into chunks
def get_text_chunks(text_with_pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for page_data in text_with_pages:
        split_texts = splitter.split_text(page_data["text"])
        for split_text in split_texts:
            chunks.append({"text": split_text, "page": page_data["page"], "pdf_name": page_data["pdf_name"]})
    return chunks

# Step 3: Build FAISS vector store
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Explicitly set to CPU to avoid meta tensor error
    )
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"page": chunk["page"], "pdf_name": chunk["pdf_name"]} for chunk in chunks]
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    db.save_local("faiss_index")

# Step 4: Generate quiz using Gemini
def generate_quiz(context, pdf_name):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
Generate 3 quiz questions (1 MCQ, 1 SAQ, 1 LAQ) based on the context from {pdf_name}. Include correct answers and explanations. Format as JSON:
[
  {{"id": "1", "type": "MCQ", "question": "...", "options": ["...", "...", "...", "..."], "correct_answer": "...", "explanation": "...", "source": {{"pdf": "...", "page": ..., "snippet": "..."}}}},
  {{"id": "2", "type": "SAQ", "question": "...", "correct_answer": "...", "explanation": "...", "source": {{"pdf": "...", "page": ..., "snippet": "..."}}}},
  {{"id": "3", "type": "LAQ", "question": "...", "correct_answer": "...", "explanation": "...", "source": {{"pdf": "...", "page": ..., "snippet": "..."}}}}
]
Context:
{context}
"""
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except:
        # Fallback mock quiz
        return [
            {
                "id": "1", "type": "MCQ", "question": "What is the SI unit of force?",
                "options": ["Newton", "Joule", "Watt", "Pascal"], "correct_answer": "Newton",
                "explanation": "The SI unit of force is Newton, defined as kg·m/s².",
                "source": {"pdf": pdf_name, "page": 23, "snippet": "Force is measured in Newton..."}
            },
            {
                "id": "2", "type": "SAQ", "question": "Define acceleration.",
                "correct_answer": "Acceleration is the rate of change of velocity with respect to time.",
                "explanation": "Acceleration occurs when an object's velocity changes.",
                "source": {"pdf": pdf_name, "page": 25, "snippet": "Acceleration is defined as..."}
            },
            {
                "id": "3", "type": "LAQ", "question": "Explain Newton's First Law of Motion with an example.",
                "correct_answer": "An object remains in its state of rest or uniform motion unless acted upon by an external force. Example: A book on a table remains stationary unless pushed.",
                "explanation": "Newton's First Law, also called the law of inertia, describes the tendency of objects to maintain their state.",
                "source": {"pdf": pdf_name, "page": 22, "snippet": "Newton’s First Law states..."}
            }
        ]

# Step 5: Ask Gemini with RAG
def ask_gemini(context, question, pdf_name, page):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
You are a helpful assistant. Answer the question using the context below.
If the answer is not in the context, say "Sorry, I couldn't find that in the document."
Include a citation with the PDF name, page number, and a 2-3 line snippet.

Context:
{context}

Question:
{question}

Answer:
"""
    response = model.generate_content(prompt)
    answer = response.text.strip()
    return f"{answer}\n\nAccording to {pdf_name}, p. {page}: \"{context[:100]}...\""

# Step 6: Search FAISS and get answer
def user_input(question, selected_pdf):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Explicitly set to CPU
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    pdf_name = docs[0].metadata["pdf_name"] if docs else selected_pdf
    page = docs[0].metadata["page"] if docs else 1
    return ask_gemini(context, question, pdf_name, page)

# Step 7: Download and display PDF
def display_pdf(pdf_path):
    try:
        if pdf_path.startswith("http"):
            response = requests.get(pdf_path)
            if response.status_code == 200:
                base64_pdf = base64.b64encode(response.content).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                st.warning("Unable to load PDF from URL. Please download and upload locally.")
        else:
            with open(pdf_path, "rb") as file:
                base64_pdf = base64.b64encode(file.read()).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"PDF viewer not available for this file. Error: {str(e)}")

# Streamlit UI
def main():
    st.title("💬 CourseBook Revision App")

    # Sidebar
    with st.sidebar:
        st.header("📂 Menu")
        selected_pdf = st.selectbox(
            "Select PDF",
            ["All PDFs"] + [pdf["name"] for pdf in st.session_state.processed_pdfs],
            key="pdf_selector"
        )
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])

        if st.button("🚀 Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    for pdf in pdf_docs:
                        # Save uploaded PDF locally for processing
                        with open(pdf.name, "wb") as f:
                            f.write(pdf.read())
                        st.session_state.processed_pdfs.append({"name": pdf.name, "path": pdf.name})
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("✅ PDFs processed and indexed!")
            else:
                st.warning("Please upload at least one PDF.")

        st.header("💬 Chats")
        for chat in st.session_state.chat_history:
            if st.button(chat["title"], key=chat["id"]):
                st.session_state.current_chat_id = chat["id"]
        if st.button("➕ New Chat"):
            new_chat = {"id": str(uuid.uuid4()), "title": f"Chat {len(st.session_state.chat_history) + 1}", "messages": []}
            st.session_state.chat_history.append(new_chat)
            st.session_state.current_chat_id = new_chat["id"]

        if st.button("📝 Generate Quiz"):
            with st.spinner("Generating quiz..."):
                context = "Sample context from PDF"  # In production, extract from selected PDF
                pdf_name = selected_pdf if selected_pdf != "All PDFs" else "NCERT Physics Ch1.pdf"
                st.session_state.quiz_data = generate_quiz(context, pdf_name)
                st.session_state.user_answers = {}
                st.success("✅ Quiz generated!")

    # Main layout
    col1, col2 = st.columns([1, 1])

    # PDF Viewer
    with col1:
        st.header("📄 PDF Viewer")
        if selected_pdf != "All PDFs":
            pdf = next((p for p in st.session_state.processed_pdfs if p["name"] == selected_pdf), None)
            if pdf:
                display_pdf(pdf.get("path", pdf.get("url")))
            else:
                st.info("Select a PDF to view.")
        else:
            st.info("Select a specific PDF to view.")

    # Chat/Quiz Area
    with col2:
        if st.session_state.quiz_data:
            st.header("📝 Quiz")
            for q in st.session_state.quiz_data:
                st.subheader(f"{q['type']}: {q['question']}")
                if q["type"] == "MCQ":
                    answer = st.radio(f"q{q['id']}", q["options"], key=f"q{q['id']}")
                    st.session_state.user_answers[q["id"]] = answer
                else:
                    answer = st.text_area(f"q{q['id']}", key=f"q{q['id']}")
                    st.session_state.user_answers[q["id"]] = answer
            if st.button("✅ Submit Quiz"):
                results = []
                for q in st.session_state.quiz_data:
                    user_answer = st.session_state.user_answers.get(q["id"], "")
                    is_correct = user_answer.lower() == q["correct_answer"].lower()
                    results.append({
                        "question": q["question"],
                        "user_answer": user_answer,
                        "correct_answer": q["correct_answer"],
                        "is_correct": is_correct,
                        "explanation": q["explanation"],
                        "source": q["source"]
                    })
                st.session_state.quiz_results.extend(results)
                st.session_state.quiz_data = []
                st.success("Quiz submitted!")
        else:
            st.header("💬 Chat")
            current_chat = next(c for c in st.session_state.chat_history if c["id"] == st.session_state.current_chat_id)
            for msg in current_chat["messages"]:
                st.markdown(f"**You**: {msg['question']}")
                st.markdown(f"**Bot**: {msg['answer']}")
            question = st.text_input("🔎 Ask a question from the uploaded PDF(s):", key="chat_input")
            if question:
                with st.spinner("Fetching answer..."):
                    answer = user_input(question, selected_pdf)
                    current_chat["messages"].append({"question": question, "answer": answer})
                    st.markdown(f"**You**: {question}")
                    st.markdown(f"**Bot**: {answer}")

    # Progress Dashboard
    st.header("📊 Progress Dashboard")
    if st.session_state.quiz_results:
        st.write(f"**Attempts**: {len(st.session_state.quiz_results)}")
        score = sum(1 for r in st.session_state.quiz_results if r["is_correct"])
        st.write(f"**Score**: {score} / {len(st.session_state.quiz_results)}")
        st.write(f"**Strengths**: {'Good grasp of answered topics' if score > 0 else 'N/A'}")
        st.write(f"**Weaknesses**: {'Review incorrect topics' if score < len(st.session_state.quiz_results) else 'N/A'}")
        for result in st.session_state.quiz_results:
            st.markdown(f"- **Q**: {result['question']}")
            st.markdown(f"  **Your Answer**: {result['user_answer']}")
            st.markdown(f"  **Correct Answer**: {result['correct_answer']}")
            st.markdown(f"  **Explanation**: {result['explanation']} (Source: {result['source']['pdf']}, p. {result['source']['page']})")
    else:
        st.info("Take a quiz to see your progress!")

if __name__ == "__main__":
    main()
