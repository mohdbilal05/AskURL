import os
import warnings
import nltk
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# =========================================================
# === SETUP PHASE ===
# =========================================================

# Load environment variables from .env (e.g., API keys)
load_dotenv()

# Suppress irrelevant warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Disable HuggingFace symlink warning (Windows-specific)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ensure that necessary NLTK tokenizers are downloaded
# These are required by Unstructured/WebBaseLoader for text processing
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# =========================================================
# === GLOBAL CONSTANTS AND VARIABLES ===
# =========================================================

# Chunk size for splitting long documents into smaller parts
CHUNK_SIZE = 800  # smaller chunks = better retrieval precision

# Pretrained embedding model for converting text into vector representations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Folder to store persistent Chroma database
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"

# Logical collection name inside the Chroma database
COLLECTION_NAME = "knowledge_base"

# Global placeholders for model and vector store (to initialize once)
llm = None
vector_store = None

# =========================================================
# === COMPONENT INITIALIZATION ===
# =========================================================

def initialize_components():
    """
    Initialize Groq LLM and Chroma vector store.
    Ensures we only load models once (singleton pattern).
    """
    global llm, vector_store

    # --- Initialize LLM (Groq) ---
    if llm is None:
        print("🔹 Initializing Groq model...")
        llm = ChatGroq(
            model="llama-3.1-8b-instant",          # Model name from Groq
            temperature=0.3,                  # Low temperature = factual answers
            max_tokens=512                    # Limit response length
        )

    # --- Initialize Vector Store (Chroma) ---
    if vector_store is None:
        print("🔹 Loading HuggingFace embeddings...")
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        # Create or load an existing Chroma vector database
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

# =========================================================
# === DATA INGESTION AND VECTOR CREATION ===
# =========================================================


def process_urls(urls):
    """Scrape and embed website content into Chroma vector DB."""
    yield "Initializing Components..."
    initialize_components()

    yield "Resetting vector store...✅"
    try:
        vector_store.reset_collection()
    except Exception:
        pass

    yield "Loading data from URLs...✅"
    loader = WebBaseLoader(
        web_paths=urls,
        requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0"}}
    )

    # ✅ Convert to Document objects
    raw_data = loader.load()

    # ✅ Safely convert every record to a valid Document
    data = []
    for d in raw_data:
        try:
            if hasattr(d, "page_content"):
                data.append(Document(page_content=d.page_content, metadata=getattr(d, "metadata", {})))
            elif isinstance(d, dict):
                page_content = d.get("page_content") or d.get("text") or str(d)
                metadata = d.get("metadata") or {}
                data.append(Document(page_content=page_content, metadata=metadata))
            else:
                data.append(Document(page_content=str(d), metadata={"source": "unknown"}))
        except Exception as e:
            print(f"⚠️ Skipping malformed document: {e}")

    if len(data) == 0:
        raise ValueError("❌ No valid documents were parsed from the provided URLs.")

    if not data:
        raise ValueError("❌ Failed to load data from the URLs. Check connectivity or parsing.")

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100,
    )
    docs = text_splitter.split_documents(data)

    if not docs:
        raise ValueError("❌ No text chunks found. Check tokenizer or page content.")

    yield f"Adding {len(docs)} chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Embedding complete!✅"

# =========================================================
# === QUERY HANDLING (RETRIEVAL + ANSWERING) ===
# =========================================================
def generate_answer(query):
    """
    Please act as an AI assistant that first retrieves the most relevant text chunks from a vector database based on a user’s query or topic.
    Then, use those retrieved text segments as context to generate a clear, accurate, and well-informed answer.
    Make sure the final response integrates the information from the text chunks smoothly, providing a coherent and contextually appropriate explanation or solution.
    Keep the tone natural and professional, and ensure the answer directly addresses the user’s question with relevant details drawn from the retrieved content.
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    print("🔍 Generating answer...")

    # Build a RetrievalQA chain that connects retriever → LLM → answer
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()  # Converts vector store into a retriever
    )

    # Run the query through the chain
    result = chain.invoke({"question": query}, return_only_outputs=True)

    # Return both answer and source references
    return result.get("answer", "No answer found."), result.get("sources", "No sources found.")

# =========================================================
# === MAIN EXECUTION (FOR STANDALONE RUN) ===
# =========================================================
if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    print("⚙️ Starting pipeline...\n")
    for step in process_urls(urls):
        print(step)

    # Example query to test retrieval accuracy
    query = "Ask questions based on the processed URL's"
    answer, sources = generate_answer(query)

    # Nicely formatted console output
    print("\n✅ DONE")
    print("──────────────────────────────")
    print(f"🔹 Query: {query}")
    print(f"💬 Answer: {answer}")
    print(f"📚 Sources: {sources}")
    print("──────────────────────────────")
