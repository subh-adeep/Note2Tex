import sys
import os

# Add the current directory to sys.path to allow imports from subdirectories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RAG.rag_implementation import build_rag_database

if __name__ == "__main__":
    print("========================================")
    print("Building RAG Index...")
    print("========================================")
    try:
        build_rag_database()
        print("✅ RAG Index built successfully.")
    except Exception as e:
        print(f"❌ Error building RAG Index: {e}")
        sys.exit(1)
