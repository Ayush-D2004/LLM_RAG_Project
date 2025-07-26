# ✅ ingest.py (refactored)
import os
import logging
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

DATA_DIR = "data"
INDEX_DIR = "index"


def load_documents(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} pages from '{file_path}'.")
    return docs


def split_documents(docs, name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✂️  Split into {len(chunks)} chunks for '{name}'.")
    return chunks


def embed_and_save(chunks, name):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    out_dir = os.path.join(INDEX_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    db.save_local(out_dir)
    print(f"✅ Vector index saved to '{out_dir}' as 'index.faiss' and 'index.pkl'.")


def process_pdfs_for_file(pdf_path):
    filename = os.path.basename(pdf_path)
    name = filename.replace(" ", "_").replace(".pdf", "")
    logging.info(f"Processing file: {filename}")
    docs = load_documents(pdf_path)
    chunks = split_documents(docs, name)
    embed_and_save(chunks, name)
    logging.info(f"Finished processing file: {filename}")
