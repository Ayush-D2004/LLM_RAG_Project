
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

import os
from huggingface_hub import login
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
- Always be descriptive and detailed in your answers.
Use the context below to answer the question. Cite sources using the format (Source: filename.pdf, page X).

Context: {context}

Question: {question}

Answer: 
""")

def load_all_indexes():
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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.getenv("GEMINI_API_KEY"))

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    return qa_chain
