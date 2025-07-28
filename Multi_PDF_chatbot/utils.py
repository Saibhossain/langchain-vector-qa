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

def extract_images_and_tables(pdf_path):
    import pdfplumber
    import fitz
    from PIL import Image
    import matplotlib.pyplot as plt
    from io import BytesIO

    visuals = []

    doc = fitz.open(pdf_path)
    for page in doc:
        for img in page.get_images(full=True):
            base_image = fitz.Pixmap(doc,img[0])
            if base_image.n< 5:
                img_data = base_image.tobytes("png")
                visuals.append(Image.open(BytesIO(img_data)))
            base_image = None
    doc.close()

    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                tables.append(table)
    return visuals, tables
