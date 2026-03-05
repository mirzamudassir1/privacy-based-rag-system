# === EDU RAG: SUPER SIMPLE OFFLINE RAG (NO OCR, NO DOWNLOADS) ===
import streamlit as st
import tempfile
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Title
st.title("EduRAG: Offline AI Study Buddy")
st.write("Upload ANY PDF → Ask → Get answers from your notes! (Works on 78MB books, no internet)")

# Clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Fix spaces
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)  # Remove junk
    return text.strip()

# Manual prompt
def create_prompt(context, question):
    return f"""Use ONLY the context below to answer the question. If not found, say "Not in your notes."

Context from notes:
{context}

Question: {question}

Answer:"""

# Upload PDF
uploaded_file = st.file_uploader("Upload your study notes (PDF)", type="pdf")

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    st.info(f"Processing {uploaded_file.size / (1024*1024):.1f} MB PDF...")

    # Load and split (handle junk)
    with st.spinner("Reading PDF..."):
        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            full_text = " ".join([doc.page_content for doc in documents])
            full_text = clean_text(full_text)  # Clean junk!
            
            if len(full_text) < 100:
                st.warning("Low text detected — PDF might be scanned or empty. Using basic search.")
                full_text = " ".join(re.findall(r'[a-zA-Z]{3,}', full_text.lower()))  # Extract words
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(full_text)
            texts = [clean_text(chunk) for chunk in chunks if len(clean_text(chunk)) > 50]  # Filter short/junk chunks
            
            if not texts:
                st.error("No usable text found. Try a text-based PDF or smaller file.")
                os.unlink(tmp_path)
                st.stop()
                
            st.success(f"Loaded {len(texts)} clean chunks from your PDF!")
        except Exception as e:
            st.error(f"PDF load failed: {e}")
            os.unlink(tmp_path)
            st.stop()

    # Clean up temp file
    os.unlink(tmp_path)

    # Build search (TF-IDF with stop words handling)
    if 'vectorizer' not in st.session_state:
        with st.spinner("Building offline search (handles junk text)..."):
            try:
                vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.95)
                st.session_state.vectorizer = vectorizer.fit(texts)
                st.session_state.tfidf_matrix = vectorizer.transform(texts)
                st.session_state.texts = texts
                st.success("Search ready! (No internet needed)")
            except ValueError as ve:
                st.error(f"Search build failed (empty text?): {ve}. Falling back to keyword search.")
                st.session_state.texts = texts  # Use simple keyword fallback

    # Ask question
    question = st.text_input("Ask a question from your notes:")
    if question:
        with st.spinner("Searching your notes..."):
            llm = OllamaLLM(model="gemma2:2b")
            
            if hasattr(st.session_state, 'tfidf_matrix'):
                # TF-IDF search
                q_vec = st.session_state.vectorizer.transform([question])
                cosines = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
                top_idx = np.argsort(cosines)[-3:][::-1]
                context = "\n\n".join([st.session_state.texts[i] for i in top_idx])
            else:
                # Keyword fallback (no TF-IDF)
                q_words = set(re.findall(r'\w+', question.lower()))
                scores = [sum(1 for word in q_words if word in text.lower()) for text in st.session_state.texts]
                top_idx = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
                context = "\n\n".join([st.session_state.texts[i] for i, _ in top_idx])
            
            prompt = create_prompt(context, question)
            answer = llm.invoke(prompt)
        
        st.subheader("**Answer from your notes:**")
        st.write(answer)
