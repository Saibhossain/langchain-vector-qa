import streamlit as st
import os
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM
from utils import load_and_index_pdfs, extract_images_and_tables

st.set_page_config(page_title="📚 PDF Q&A Assistant", layout="wide")

st.sidebar.title("📤 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🤖 Intelligent PDF Q&A Assistant")
st.markdown("Ask anything about the uploaded PDFs.")

if uploaded_files:
    with st.spinner("🔍 Processing PDFs..."):
        os.makedirs("data", exist_ok=True)
        temp_paths = []
        for f in uploaded_files:
            path = os.path.join("/Users/mdsaibhossain/code/python/PDF_Q&A_Chatbot_usingLangChain/Multi_PDF_chatbot/data", f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            temp_paths.append(path)

        vectorstore = load_and_index_pdfs(temp_paths)
        retriever = vectorstore.as_retriever()

        llm = OllamaLLM(model="gemma3n:latest")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


    query = st.text_input("🔎 Ask a question")
    if query:
        with st.spinner("🧠 Thinking..."):
            result = qa_chain({"query": query})
            st.session_state.history.append((query, result["result"]))
            st.success("✅ Answer ready!")

    if st.session_state.history:
        st.subheader("💬 Chat History")
        for q, a in st.session_state.history[::-1]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")


    st.subheader("📊 Visual Elements from PDFs")
    for path in temp_paths:
        st.markdown(f"**📁 {os.path.basename(path)}**")
        images, tables = extract_images_and_tables(path)

        # Show images (charts/diagrams)
        for img in images[:3]:  # limit for performance
            st.image(img, caption="Chart or Diagram", use_column_width=True)

        # Show tables
        for table in tables[:2]:  # limit for performance
            if table:
                st.table(table)
else:
    st.info("👈 Please upload PDF files from the sidebar.")