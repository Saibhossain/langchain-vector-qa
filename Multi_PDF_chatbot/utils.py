import os

from gitdb.fun import chunk_size
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_and_index_pdfs(pdf_file):
    all_docs =[]
    for pdf in pdf_file:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)

        embadder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = FAISS.from_documents(chunks,embadder)
        return vectorstore

