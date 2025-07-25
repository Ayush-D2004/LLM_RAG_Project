import streamlit as st
from streamlit_option_menu import option_menu
import os
import time
import logging
import random
import shutil
from ingest import process_pdfs_for_file
from qa_pipeline import load_vectorstore, get_top_k_context, query_gemini
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

import warnings
warnings.filterwarnings("ignore") 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Copilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LIGHT THEME STYLING WITH DYNAMIC ELEMENTS ---
def load_css():
    st.markdown("""
    <style>
        /* Base light theme */
        .stApp {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
            color: #333333;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #ffffff 0%, #f8f9fa 100%);
            border-right: 1px solid #e0e0e0;
            padding: 1.5rem 1rem;
        }
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #2c3e50;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(90deg, #3498db, #2c3e50);
            border: none;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            background: linear-gradient(90deg, #2980b9, #1a2530);
        }

        /* Enhanced chat bubbles */
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        .chat-bubble {
            padding: 1.2rem 1.5rem;
            border-radius: 20px;
            margin-bottom: 1.2rem;
            max-width: 85%;
            box-shadow: 0 3px 10px rgba(0,0,0,0.08);
            position: relative;
            animation: fadeIn 0.3s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user-bubble {
            background: linear-gradient(135deg, #3498db, #2c3e50);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 5px;
        }
        .assistant-bubble {
            background: linear-gradient(135deg, #f1f2f6, #e2e8f0);
            color: #2d3436;
            align-self: flex-start;
            border-bottom-left-radius: 5px;
        }
        .chat-icon {
            font-size: 1.5rem;
            margin-right: 0.8rem;
        }
        .bubble-content {
            line-height: 1.5;
        }

        /* Thinking animation */
        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.7; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(1); opacity: 0.7; }
        }
        .thinking-animation {
            display: flex;
            align-items: center;
            justify-content: center;
            font-style: italic;
            color: #7f8c8d;
            margin-top: 1rem;
            animation: pulse 2s infinite;
        }

        /* Navigation bar */
        .streamlit-option-menu {
            background: white !important;
            border-bottom: 1px solid #e0e0e0;
        }
        .streamlit-option-menu .nav-link {
            color: #7f8c8d !important;
            transition: all 0.3s;
            font-weight: 500;
        }
        .streamlit-option-menu .nav-link:hover {
            color: #2c3e50 !important;
            background-color: #f8f9fa !important;
        }
        .streamlit-option-menu .nav-link-selected {
            background-color: #e3f2fd !important;
            color: #2c3e50 !important;
        }

        /* Main header */
        .main-header {
            text-align: center;
            margin: 1.5rem 0 2rem;
            animation: fadeInDown 0.8s ease-out;
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .main-header h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #3498db, #2c3e50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .main-header p {
            font-size: 1.2rem;
            color: #7f8c8d;
            max-width: 700px;
            margin: 0 auto;
        }

        /* File uploader */
        .stFileUploader {
            background-color: #ffffff;
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s;
        }
        .stFileUploader:hover {
            border-color: #2980b9;
            background-color: #f8f9fa;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #3498db, #2c3e50);
        }
        
        /* Toast notifications */
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            animation: slideIn 0.4s, fadeOut 0.5s 2.5s forwards;
        }
        @keyframes slideIn {
            from {right: -300px; opacity: 0;}
            to {right: 20px; opacity: 1;}
        }
        @keyframes fadeOut {
            from {opacity: 1;}
            to {opacity: 0;}
        }
        .toast-success { background: linear-gradient(90deg, #2ecc71, #27ae60); }
        .toast-warning { background: linear-gradient(90deg, #f39c12, #d35400); }
        .toast-error { background: linear-gradient(90deg, #e74c3c, #c0392b); }
        
        /* Stats cards */
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: transform 0.3s;
            border: 1px solid #e0e0e0;
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            font-size: 2.5rem;
            margin: 0.5rem 0;
            color: #3498db;
        }
        .stat-card p {
            color: #7f8c8d;
            margin: 0;
        }
        
        /* Uploaded file card */
        .file-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
            position: relative;
        }
        .file-card .remove-btn {
            position: absolute;
            top: 5px;
            right: 10px;
            background: none;
            border: none;
            color: #e74c3c;
            cursor: pointer;
            font-size: 1.2rem;
        }
        .file-card .remove-btn:hover {
            color: #c0392b;
        }
        .file-card .file-name {
            margin: 0;
            font-weight: 500;
            color: #2c3e50;
        }
        .file-card .file-size {
            margin: 0;
            font-size: 0.85rem;
            color: #7f8c8d;
        }
    </style>
    """, unsafe_allow_html=True)

def show_toast(message, type="success"):
    toast_class = f"toast toast-{type}"
    st.markdown(f"""
        <div class="{toast_class}">
            {message}
        </div>
    """, unsafe_allow_html=True)
    time.sleep(3)

def get_stats():
    """Get processing statistics"""
    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        return len(files)
    return 0

def format_file_size(size_in_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} TB"

def remove_file(file_name):
    """Remove file from data directory and its corresponding index folder"""
    try:
        # Remove from data directory
        data_file_path = os.path.join("data", file_name)
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
        
        # Remove corresponding index folder (assuming it's named after the file without extension)
        file_base_name = os.path.splitext(file_name)[0]
        index_folder_path = os.path.join("index", file_base_name)
        if os.path.exists(index_folder_path):
            shutil.rmtree(index_folder_path)
            
        return True
    except Exception as e:
        logging.error(f"Error removing file {file_name}: {e}")
        return False

load_css()

# --- SESSION STATE INITIALIZATION ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🚀 Enterprise Copilot")
    st.markdown("---")

    st.header("📁 Document Management")

    # Optional: force reprocessing toggle
    force_reprocess = st.checkbox("🔁 Force reprocess documents", value=False)

    # Debug info toggle
    show_debug = st.checkbox("Show debug info", value=True)

    # File uploader comes first!
    uploaded_files = st.file_uploader(
        "Upload company documents",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # Ensure processed_files is a set
    if "processed_files" not in st.session_state or not isinstance(st.session_state.processed_files, set):
        st.session_state.processed_files = set()

    # --- AUTO-INDEXING: Always scan data/ for PDFs and process if needed ---
    data_dir = "data"
    index_dir = "index"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    debug_msgs = []
    # Process any new uploads
    if uploaded_files:
        for uploaded_file in uploaded_files:
            safe_filename = uploaded_file.name.replace(" ", "_")
            file_path = os.path.join(data_dir, safe_filename)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                debug_msgs.append(f"Saved file: {safe_filename}")
    # Scan data/ for PDFs and process if not indexed or if force_reprocess
    processed_files = []
    skipped_files = []
    errors = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.pdf'):
            index_name = os.path.splitext(file_name)[0]
            index_path = os.path.join(index_dir, index_name)
            file_path = os.path.join(data_dir, file_name)
            needs_processing = force_reprocess or not os.path.exists(index_path) or not os.path.exists(os.path.join(index_path, "index.faiss"))
            if needs_processing:
                try:
                    process_pdfs_for_file(file_path)
                    processed_files.append(file_name)
                    debug_msgs.append(f"Processed file: {file_name}")
                except Exception as e:
                    errors.append(file_name)
                    debug_msgs.append(f"Error processing {file_name}: {e}")
            else:
                skipped_files.append(file_name)
                debug_msgs.append(f"Skipped file (already indexed): {file_name}")
            st.session_state.processed_files.add(file_name)
    # After processing, always try to load the vectorstore
    try:
        st.session_state.db = load_vectorstore(index_dir)
        st.session_state.processed = True
        debug_msgs.append("Vectorstore loaded successfully.")
        num_indexes = len(st.session_state.db)
        num_docs = 0
        for db in st.session_state.db:
            try:
                num_docs += db.index.ntotal
            except Exception:
                pass
        debug_msgs.append(f"Loaded {num_indexes} index(es), total {num_docs} documents.")
    except Exception as e:
        st.session_state.db = None
        st.session_state.processed = False
        debug_msgs.append(f"Failed to load vectorstore: {e}")
    # Toasts
    if processed_files:
        show_toast(f"✅ Processed {len(processed_files)} new document(s).", "success")
    if skipped_files:
        show_toast(f"ℹ️ Skipped {len(skipped_files)} already-processed document(s).", "warning")
    if errors:
        show_toast(f"❌ Failed to process: {', '.join(errors)}", "error")
    # Show debug messages in sidebar for troubleshooting
    if show_debug:
        st.sidebar.markdown("### 🐞 Debug Info")
        for msg in debug_msgs:
            st.sidebar.markdown(f"- {msg}")
        # Add a debug section to show session state for troubleshooting
        st.markdown("---")
        st.markdown("### 🐞 Session State Debug")
        # Show actual objects, not stringified
        st.json({k: v for k, v in st.session_state.items()})
        # Show number of loaded indexes and total docs if db is loaded
        if st.session_state.get("db"):
            num_indexes = len(st.session_state.db)
            num_docs = 0
            for db in st.session_state.db:
                try:
                    num_docs += db.index.ntotal
                except Exception:
                    pass
            st.markdown(f"**Indexes loaded:** {num_indexes}")
            st.markdown(f"**Total docs in vectorstore:** {num_docs}")
        # Show contents of data/ and index/ for troubleshooting
        st.markdown("---")
        st.markdown("### 📂 Data Directory Contents")
        try:
            st.json(os.listdir("data"))
        except Exception:
            st.write("data/ directory not found")
        st.markdown("### 📂 Index Directory Contents")
        try:
            st.json({d: os.listdir(os.path.join("index", d)) for d in os.listdir("index")})
        except Exception:
            st.write("index/ directory not found")

    # Display uploaded files with remove option
    st.markdown("---")
    st.header("📄 Uploaded Documents")
    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        if files:
            for file_name in files:
                file_path = os.path.join(data_dir, file_name)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"""
                    <div class="file-card">
                        <p class="file-name">📄 {file_name}</p>
                        <p class="file-size">{format_file_size(file_size)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Using a form to handle the remove action
                with col2:
                    if st.button("❌", key=f"remove_{file_name}", help="Remove this document"):
                        if remove_file(file_name):
                            show_toast(f"🗑️ Removed {file_name}", "success")
                        else:
                            show_toast(f"❌ Failed to remove {file_name}", "error")
        else:
            st.info("No documents uploaded yet")
    else:
        st.info("No documents uploaded yet")

# --- TOP NAVIGATION BAR ---
selected = option_menu(
    menu_title=None,
    options=["Assistant", "Dashboard"],
    icons=["chat-dots-fill", "speedometer2"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "white"},
        "icon": {"color": "#3498db", "font-size": "16px"},
        "nav-link": {
            "font-size": "15px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#f8f9fa",
        },
        "nav-link-selected": {"background-color": "#e3f2fd"},
    },
)

# --- ASSISTANT PAGE ---
if selected == "Assistant":
    st.markdown('<div class="main-header"><h1>🚀 Enterprise Copilot</h1><p>Your AI-powered business assistant</p></div>', unsafe_allow_html=True)

    # Add a debug section to show session state for troubleshooting
    st.markdown("---")
    st.markdown("### 🐞 Session State Debug")
    st.json({k: str(v) for k, v in st.session_state.items()})
    # Show number of loaded indexes and total docs if db is loaded
    if st.session_state.get("db"):
        num_indexes = len(st.session_state.db)
        num_docs = 0
        for db in st.session_state.db:
            try:
                num_docs += db.index.ntotal
            except Exception:
                pass
        st.markdown(f"**Indexes loaded:** {num_indexes}")
        st.markdown(f"**Total docs in vectorstore:** {num_docs}")
    # Show contents of data/ and index/ for troubleshooting
    st.markdown("---")
    st.markdown("### 📂 Data Directory Contents")
    try:
        st.json(os.listdir("data"))
    except Exception:
        st.write("data/ directory not found")
    st.markdown("### 📂 Index Directory Contents")
    try:
        st.json({d: os.listdir(os.path.join("index", d)) for d in os.listdir("index")})
    except Exception:
        st.write("index/ directory not found")

    # If db is not loaded, show a clear error
    if not st.session_state.get("db"):
        st.warning("Knowledge engine is not loaded. Please upload and process at least one document.")

    # Chat display
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        if st.session_state.messages:
            for message in st.session_state.messages:
                role = message["role"]
                icon = "👤" if role == "user" else "🤖"
                bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
                st.markdown(f"""
                <div class="chat-bubble {bubble_class}">
                    <span class="chat-icon">{icon}</span>
                    <div class="bubble-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-bubble assistant-bubble">
                <span class="chat-icon">🤖</span>
                <div class="bubble-content">Hello! I'm your Enterprise Copilot. Once you've processed your documents, ask me anything about your company's knowledge base.</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("💬 Ask me anything about your documents...", key="user_chat_input"):
        if st.session_state.processed:
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Trigger assistant response immediately
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("""
            <div class="thinking-animation">
                <span class="chat-icon">🚀</span>
                <span>Enterprise Copilot is analyzing...</span>
            </div>
            """, unsafe_allow_html=True)

            try:
                context_string, source_docs = get_top_k_context(prompt, st.session_state.db)
                if not context_string.strip():
                    warning_msg = "I couldn't find relevant information in the documents."
                    st.session_state.messages.append({"role": "assistant", "content": warning_msg})
                    thinking_placeholder.empty()
                else:
                    full_response = query_gemini(context_string, prompt)
                    if not full_response or not full_response.strip():
                        full_response = "❌ No response generated by the AI model."
                    logging.info(f"User prompt received: {prompt}")
                    logging.info(f"Context string: {context_string}")
                    logging.info(f"Full response: {full_response}")
                    logging.info(f"Source documents: {source_docs}")

                    full_response += "\n\n---\n**Sources:**\n"
                    unique_sources = set()
                    for d in source_docs:
                        file = os.path.basename(d.metadata.get("source", "Unknown"))
                        page = d.metadata.get("page", "?")
                        unique_sources.add(f"- `{file}`, Page {page}")

                    full_response += "\n".join(sorted(list(unique_sources)))
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    thinking_placeholder.empty()
            except Exception as e:
                logging.error(f"Answer generation error: {e}")
                error_msg = f"An error occurred while generating the answer: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                show_toast("❌ Failed to get answer", "error")
                thinking_placeholder.empty()
        else:
            show_toast("⚠️ Please process your documents first!", "warning")

    

# --- DASHBOARD PAGE ---
elif selected == "Dashboard":
    st.markdown('<div class="main-header"><h1>📊 Enterprise Copilot Dashboard</h1><p>Insights and analytics for your knowledge base</p></div>', unsafe_allow_html=True)
    
    # Stats cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{get_stats()}</h3>
            <p>Documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{len(st.session_state.messages) // 2 if st.session_state.messages else 0}</h3>
            <p>Interactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology stack
    st.subheader("🛠️ Powered by Cutting-Edge Technology")
    
    tech_cols = st.columns(5)
    technologies = [
        ("Streamlit", "🌐"),
        ("LangChain", "🔗"),
        ("Ollama", "🦙"),
        ("FAISS", "🔍"),
        ("MiniLM", "🔤")
    ]
    
    for i, (tech, icon) in enumerate(technologies):
        with tech_cols[i]:
            st.markdown(f"""
            <div style="background:white; border-radius:10px; padding:1.5rem 1rem; text-align:center; box-shadow:0 2px 5px rgba(0,0,0,0.05);">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:600;">{tech}</div>
            </div>
            """, unsafe_allow_html=True)