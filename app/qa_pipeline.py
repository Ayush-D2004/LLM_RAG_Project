import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from logger import logging

INDEX_DIR = "./index"

custom_prompt = PromptTemplate.from_template("""
You are an intelligent AI assistant trained to answer questions based strictly on internal company documents.

Guidelines:
- ONLY answer from the given context.
- If the context is insufficient, say: "I couldn't find relevant information in the documents."
- Always cite sources like this: (Source: filename.pdf, Page X).
- Be concise, clear, and professional in tone.
- If the question is ambiguous or unclear, ask a clarifying follow-up.
- Be brief and to the point while explaining to the user.

Context: {context}

Question: {question}

Answer:
""")

def load_all_indexes():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstores = []

    for folder in os.listdir(INDEX_DIR):
        subdir = os.path.join(INDEX_DIR, folder)
        if os.path.isdir(subdir):
            try:
                db = FAISS.load_local(subdir, embeddings, allow_dangerous_deserialization=True)
                vectorstores.append(db)
                logging.info(f"Loaded index from '{subdir}'")
            except Exception as e:
                logging.error(f"Skipped '{subdir}': {e}")

    if not vectorstores:
        raise ValueError("No valid FAISS indexes found in the index directory.")

    # Merge all vectorstores into one
    merged_db = vectorstores[0]
    for db in vectorstores[1:]:
        merged_db.merge_from(db)
    logging.info(f"Merged {len(vectorstores)} indexes into a single database.")

    return merged_db.as_retriever(search_kwargs={"k": 4})


def load_rag_chain():
    retriever = load_all_indexes()
    llm = Ollama(model="llama3")
    logging.info(f"Loaded Ollama model 'llama3'.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    logging.info(f"Loaded RAG pipeline successfully.")

    return qa_chain
