import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Missing GEMINI_API_KEY in your environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- PROMPT FORMATTING ---
def format_prompt(context: str, query: str) -> str:
    """Creates a structured prompt for the Gemini model."""
    return f"""
You are a helpful AI assistant for internal company knowledge.
Your guidelines are:
- You must ONLY answer using the provided context.
- If the context does not contain the answer, you must state: "I couldn't find relevant information in the documents."
- Your tone should be concise, clear, and professional.
- If the user's question is unclear, ask a clarifying question.
- All answers must be brief and to the point.

Use the context below to answer the question.

Context:
{context}

Question: {query}

Answer:
"""

# --- VECTORSTORE OPERATIONS ---
def load_vectorstore(index_dir: str) -> list:
    """Loads all valid FAISS indexes from a directory."""
    vector_dbs = []
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists(index_dir):
        logger.warning(f"Index directory '{index_dir}' not found.")
        return []

    for subdir in os.listdir(index_dir):
        path = os.path.join(index_dir, subdir)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "index.faiss")):
            try:
                db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
                vector_dbs.append(db)
                logger.info(f"✅ Loaded index from '{path}'")
            except Exception as e:
                logger.error(f"⚠️ Failed to load index from '{path}': {e}")

    if not vector_dbs:
        raise ValueError("❌ No valid FAISS indexes were found or loaded.")

    return vector_dbs

def get_top_k_context(query: str, dbs: list, k: int = 4) -> tuple[str, list]:
    """Retrieves the most relevant context and the source documents."""
    if not dbs:
        return "", []

    merged_db = dbs[0]
    if len(dbs) > 1:
        for db in dbs[1:]:
            merged_db.merge_from(db)

    retriever = merged_db.as_retriever(search_kwargs={"k": k})
    source_docs = retriever.get_relevant_documents(query)
    
    context_text = "\n\n".join([doc.page_content for doc in source_docs])
    
    return context_text, source_docs

# --- GENERATIVE AI QUERY ---
def query_gemini(context: str, query: str):
    """Queries the Gemini model with the provided context and question."""
    try:
        model = genai.GenerativeModel('gemini-pro')

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        full_prompt = format_prompt(context, query)
        
        response = model.generate_content(
            full_prompt,
            safety_settings=safety_settings
        )

        return response.text

    except Exception as e:
        logger.error(f"❌ An unexpected Gemini API error occurred: {e}")
        return "❌ An error occurred while communicating with the AI model."