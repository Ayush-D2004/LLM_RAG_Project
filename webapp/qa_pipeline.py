
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

INDEX_DIR = "index"

custom_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant for internal company knowledge.
Guidelines:
- ONLY answer from the given context.
- If the context is insufficient, say: "I couldn't find relevant information in the documents."
- Always cite sources like this: (Source: filename.pdf, Page X).
- Be concise, clear, and professional in tone.
- If the question is ambiguous or unclear, ask a clarifying follow-up.
- Be brief and to the point while explaining to the user.
- Always end with a question asking to follow up.
- If user greets you then make a polite greetings and ask how can I help you.

Use the context below to answer the question. Cite sources using the format (Source: filename.pdf, page X).

Context: {context}

Question: {question}

Answer: 
""")

def load_all_indexes():
    """Load all FAISS indexes from the index directory."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dbs = []

    for name in os.listdir(INDEX_DIR):
        dir_path = os.path.join(INDEX_DIR, name)
        if os.path.isdir(dir_path):
            try:
                db = FAISS.load_local(dir_path, embeddings, allow_dangerous_deserialization=True)
                dbs.append(db)
                print(f"✅ Loaded index from '{dir_path}'")
            except Exception as e:
                print(f"⚠️ Skipped '{dir_path}': {e}")

    if not dbs:
        raise RuntimeError("❌ Failed to load RAG pipeline: No valid FAISS indexes found.")

    # Merge all DBs into one
    merged_db = dbs[0]
    for db in dbs[1:]:
        merged_db.merge_from(db)

    retriever = merged_db.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="llama3")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    return qa_chain
