# ✅ ingest.py (refactored)
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = "data"
INDEX_DIR = "index"


def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} pages from '{pdf_path}'.")
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
    docs = load_documents(pdf_path)
    chunks = split_documents(docs, name)
    embed_and_save(chunks, name)
