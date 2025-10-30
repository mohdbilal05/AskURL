# 🧠 AskURL — Chat with Any Website using AI

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen?logo=streamlit)](https://askurl-ai.streamlit.app/)
[![GitHub](https://img.shields.io/badge/View%20Code-GitHub-black?logo=github)](https://github.com/mohdbilal05/AskURL)
[![Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-orange?logo=chainlink)]
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## 🚀 Overview

**AskURL** is an end-to-end **Retrieval-Augmented Generation (RAG)** web app that lets you **chat with any website**.  
Just enter a URL — the app reads, processes, and extracts information, then uses an **LLM** to answer your questions contextually.  

🔗 **Live App:** [https://askurl-ai.streamlit.app/](https://askurl-ai.streamlit.app/)  
💻 **GitHub Repo:** [https://github.com/mohdbilal05/AskURL](https://github.com/mohdbilal05/AskURL)

---

## 💡 Key Features

✅ Chat with any webpage in real-time  
✅ Dynamic web scraping and text cleaning  
✅ Vector-based document retrieval using **ChromaDB**  
✅ Contextual query answering via **LangChain + Groq LLMs**  
✅ Fully deployed on **Streamlit Cloud**  
✅ Modular and scalable architecture for GenAI workflows  

---

## ⚙️ Tech Stack

| Category | Tools & Frameworks |
|-----------|--------------------|
| **Frontend & Deployment** | Streamlit Cloud |
| **LLM Framework** | LangChain, LangChain-Groq, LangChain-HuggingFace |
| **Vector Database** | ChromaDB |
| **Embeddings & NLP** | Sentence-Transformers, Transformers, NLTK |
| **Model Acceleration** | Torch, Accelerate |
| **Utilities** | BeautifulSoup4, Python-dotenv, Pandas, NumPy, TQDM |

---

## 🧩 System Workflow

1. **URL Input:** User provides any valid webpage URL.  
2. **Content Extraction:** The app scrapes and preprocesses text using BeautifulSoup.  
3. **Chunking & Embedding:** Sentences are converted into vector embeddings.  
4. **Vector Storage:** Embeddings are stored in **ChromaDB** for retrieval.  
5. **Context Retrieval:** LangChain retrieves relevant chunks for each query.  
6. **Response Generation:** Groq LLM produces contextual answers.  
7. **Display:** Streamlit UI displays the response in a conversational format.  

---

## 🧠 Project Structure

<img width="562" height="271" alt="image" src="https://github.com/user-attachments/assets/91bef67a-8a3a-407f-81f4-e254a9c5ba97" />



---

## 🧰 Installation & Setup (Local)

```bash
# 1. Clone the repository
git clone https://github.com/mohdbilal05/AskURL.git
cd AskURL

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # for macOS/Linux
venv\Scripts\activate         # for Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your environment variables
# Create a .env file and add:
# GROQ_API_KEY=your_groq_api_key

# 5. Run the app
streamlit run app.py

