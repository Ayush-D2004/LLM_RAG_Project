from qa_pipeline import load_rag_chain
from logger import logging
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    try:
        chain = load_rag_chain()
    except Exception as e:
        print(f"Failed to load RAG pipeline: {e}")
        logging.error(f"Failed to load RAG pipeline: {e}")
        return

    print("\n Enterprise Copilot is ready. Type your question or 'exit' to quit.")
    logging.info("\n Enterprise Copilot is engaging with user. Everything working fine")

    try:
        while True:
            query = input("\n🧑 You: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("👋 Exiting. Goodbye!")
                break

            result = chain.invoke({"query": query})
            answer = result.get('result', 'No answer returned.')
            print(f"\n🤖 Copilot :\n{answer}\n")

            # Show citations (source documents)
            print("📚 Sources:")
            for doc in result.get('source_documents', []):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                print(f"  - (Source: {source}, Page: {page})")

    except KeyboardInterrupt:
        print("\n👋 Interrupted. Exiting...")

if __name__ == "__main__":
    main()
