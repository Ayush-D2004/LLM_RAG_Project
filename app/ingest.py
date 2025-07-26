from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from logger import logging


DATA_DIR = "data"
OUTPUT_ROOT = "index"

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
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} pages from '{os.path.basename(file_path)}'.")
    return documents

def split_documents(docs, filename):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logging.info(f"Split into {len(chunks)} chunks for '{filename}'.")
    return chunks

def embed_and_save(chunks, pdf_filename):
    base_name = os.path.splitext(pdf_filename)[0]
    output_dir = os.path.join(OUTPUT_ROOT, base_name)  # Changed from f"{base_name}_index" to just base_name
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Embedding and saving vector index for '{base_name}' to '{output_dir}'.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(output_dir)
    logging.info(f"Embeddings and vector index saved to '{output_dir}'.")

    # Verify files exist but don't rename them
    index_faiss = os.path.join(output_dir, "index.faiss")
    index_pkl = os.path.join(output_dir, "index.pkl")

    if os.path.exists(index_faiss) and os.path.exists(index_pkl):
        logging.info(f"Vector index saved to '{output_dir}' as 'index.faiss' and 'index.pkl'.")
    else:
        logging.error(f"Expected FAISS output files were not found after saving in '{output_dir}'.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    supported_exts = [".pdf", ".docx", ".txt"]
    data_files = [f for f in os.listdir(DATA_DIR) if os.path.splitext(f)[1].lower() in supported_exts]

    if not data_files:
        raise FileNotFoundError("No supported files (.pdf, .docx, .txt) found in 'data' directory.")

    logging.info(f"Found {len(data_files)} supported file(s) in '{DATA_DIR}'.\n")

    for data_file in data_files:
        path = os.path.join(DATA_DIR, data_file)
        docs = load_documents(path)
        chunks = split_documents(docs, data_file)
        embed_and_save(chunks, data_file)
    
    logging.info("Ingest.py completed with no errors.")
