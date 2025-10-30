import streamlit as st
from rag import process_urls, generate_answer, initialize_components

# === Streamlit UI Setup ===
import streamlit as st

st.set_page_config(
    page_title="AskURL - Your AI Web Research Assistant",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 AskURL — Your AI Web Research Assistant")
st.markdown("""
#### Get instant answers from any webpage, article, or online document.
Just paste a URL, and let AI read, analyze, and summarize the content for you.  
Then, ask **any question** — and get accurate, context-aware responses instantly.
""")
st.divider()


# Initialize persistent state for components
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "urls_processed" not in st.session_state:
    st.session_state.urls_processed = False

# Sidebar URL inputs
st.sidebar.header("🔗 Enter Article URLs")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")
process_url_button = st.sidebar.button("⚙️ Process URLs")

status_placeholder = st.empty()

# === URL Processing ===
if process_url_button:
    urls = [u.strip() for u in (url1, url2, url3) if u.strip()]
    if not urls:
        st.warning("⚠️ You must provide at least one URL.")
    else:
        try:
            with st.spinner("🔍 Processing URLs... please wait."):
                for step in process_urls(urls):
                    status_placeholder.info(step)
            st.session_state.urls_processed = True
            from rag import vector_store, llm
            st.session_state.vector_store = vector_store
            st.session_state.llm = llm
            st.success("✅ URLs processed successfully!")
        except Exception as e:
            st.error(f"❌ Error during URL processing: {e}")
            st.session_state.urls_processed = False

st.divider()

# === Query Section ===
query = st.text_input("💬 Ask a Question", placeholder="e.g., What was the 30-year fixed mortgage rate and its date?")
submit = st.button("🚀 Submit Question")

if submit:
    if not st.session_state.urls_processed:
        st.warning("⚠️ Please process URLs before asking a question.")
    elif not query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        try:
            # Reconnect vector store and LLM from session state
            import rag
            rag.vector_store = st.session_state.vector_store
            rag.llm = st.session_state.llm

            with st.spinner("🤖 Generating answer..."):
                answer, sources = generate_answer(query)
            st.header("✅ Answer")
            st.write(answer)
            if sources:
                st.subheader("📚 Sources")
                for s in sources.split("\n"):
                    if s.strip():
                        st.write(f"🔗 [{s}]({s})")
        except Exception as e:
            st.error(f"⚠️ {e}")

st.divider()
st.caption("💡 Built with LangChain, Groq, and HuggingFace — © Real Estate Assistant 2025")
