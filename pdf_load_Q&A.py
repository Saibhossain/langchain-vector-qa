from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama  # Use this if running locally
from sqlalchemy.dialects.mysql.mariadb import loader


# Load PDF
pdf_path ="dip_paper (1).pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages from PDF.")

splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} chunks.")

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vectorstore = FAISS.from_documents(chunks,embedding_model)
print("✅ FAISS vectorstore created with embedded chunks.")
