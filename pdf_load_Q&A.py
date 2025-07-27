from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM # Use this if running locally
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

llm = OllamaLLM(model="gemma3n:latest")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    return_source_documents=True
)

while True:
    query = input("📘 Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa_chain({"query": query})
    print("\n🧠 Answer:")
    print(result["result"])

    # Optional: Show source pages
    print("\n📄 Source Snippet:")
    for doc in result["source_documents"][:1]:
        print(doc.page_content[:300])
        print("-" * 50)