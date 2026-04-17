from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from groq import Groq
from langdetect import detect
from deep_translator import GoogleTranslator
import faiss
import numpy as np
import os
import pickle

from dotenv import load_dotenv
import os

load_dotenv() 

from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

from docx import Document
from pptx import Presentation


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


print("Loading models...")
embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")


FAISS_FILE = "faiss_index.index"
DOCS_FILE = "docs.pkl"
FILES_RECORD = "files_record.pkl"


def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def translate_answer(text, target_lang):
    try:
        if target_lang == "Sinhala":
            return GoogleTranslator(source='en', target='si').translate(text)
        elif target_lang == "Tamil":
            return GoogleTranslator(source='en', target='ta').translate(text)
        else:
            return text
    except:
        return text


def detect_language(text):
    try:
        lang = detect(text)
        if lang == "si":
            return "Sinhala"
        elif lang == "ta":
            return "Tamil"
        else:
            return "English"
    except:
        return "English"


chat_history = []
MAX_MEMORY = 5

def update_chat_history(q, a):
    chat_history.append((q, a))
    if len(chat_history) > MAX_MEMORY:
        chat_history.pop(0)

def get_chat_history():
    return "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])



def load_pdf(path):
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        return [d.page_content for d in docs]
    except Exception as e:
        print(" Error loading PDF:", path)
        print("Reason:", e)
        return []

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [f.read()]

def load_docx(path):
    doc = Document(path)
    return ["\n".join([p.text for p in doc.paragraphs])]

def load_pptx(path):
    prs = Presentation(path)
    text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return ["\n".join(text)]

def load_file(path):
    ext = path.lower().split(".")[-1]

    if ext == "pdf":
        return load_pdf(path)
    elif ext == "txt":
        return load_txt(path)
    elif ext == "docx":
        return load_docx(path)
    elif ext == "pptx":
        return load_pptx(path)
    else:
        return []


def build_index(paths):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = []

    for path in paths:
        contents = load_file(path)

        for content in contents:
            translated_content = translate_to_english(content)
            chunks = splitter.split_text(translated_content)

            for c in chunks:
                docs.append({
                    "text": c,
                    "source": path
                })

    texts = [d["text"] for d in docs]

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    emb_array = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(emb_array.shape[1])
    index.add(emb_array)

    return index, docs


pdf_folder = "pdfs"

paths = []

for file_name in os.listdir(pdf_folder):
    file_path = os.path.join(pdf_folder, file_name)

    if os.path.isfile(file_path) and file_name.lower().endswith(".pdf"):
        print("Loading file:", file_name)
        paths.append(file_path)


if os.path.exists(FAISS_FILE) and os.path.exists(DOCS_FILE):

    print("Loading saved FAISS index...")

    index = faiss.read_index(FAISS_FILE)

    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)

    print("Loaded successfully. Vectors:", index.ntotal)

else:
    print(" Index not found. Building index FIRST TIME...")

    index, docs = build_index(paths)

    faiss.write_index(index, FAISS_FILE)

    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

    print(" Index built and saved successfully")



def search_web(query):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return "\n".join([r["body"] for r in results])



def rag_pipeline(question):

    q_emb = embedding_model.encode([question])
    q_emb = np.array(q_emb).astype("float32")

    distances, indices = index.search(q_emb, 5)

    candidates = [docs[i] for i in indices[0]]
    top_chunks = candidates[:3]

    context = "\n".join([c["text"] for c in top_chunks])

    best_score = distances[0][0]

    

    if best_score < 0.8:
        final_context = context

    elif best_score < 1.2:
        web_data = search_web(question)
        final_context = context + "\n\n" + web_data

    else:
        final_context = search_web(question)

   


    answer = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a highly accurate assistant. Always give correct answers."},
            {"role": "user", "content": final_context + "\n\n" + question}
        ]
    )

    return answer.choices[0].message.content





if __name__ == "__main__":

    while True:

        question = input("\nAsk (or type exit / clear): ")

        if question.lower() == "exit":
            break

        if question.lower() == "clear":
            chat_history.clear()
            print("Chat history cleared!")
            continue

        q_emb = embedding_model.encode([question])
        q_emb = np.array(q_emb).astype("float32")

        distances, indices = index.search(q_emb, 5)

        candidates = [docs[i] for i in indices[0]]

        top_chunks = candidates[:3]

        context = "\n".join([c["text"] for c in top_chunks])

        if not context.strip():
            print(" No data in PDFs. Searching internet...")
            context = search_web(question)


        def is_context_useful(distances):
            return distances[0][0] < 0.8

        best_score = distances[0][0]

        use_pdf = best_score < 0.8


        if use_pdf:

            print("\n Sources:")
            for c in top_chunks:
                print("-", c["source"])

            answer = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Answer using context if available. Otherwise answer normally."},
                    {"role": "user", "content": context + "\n\n" + question}
                ]
            )

            print("\n Answer:\n", answer.choices[0].message.content)

        else:

            answer = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer normally if context is not useful."},
                    {"role": "user", "content": question}
                ]
            )

            print("\n Answer:\n", answer.choices[0].message.content)

        update_chat_history(question, answer.choices[0].message.content)
