from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step 1: Load PDFs
docs_path = "data"
all_documents = []
for file in os.listdir(docs_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(docs_path, file))
        all_documents.extend(loader.load())

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(all_documents)

# Step 3: Embed and store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedding_model)
db.save_local("vectorstore")

print(f"✅ Ingested and saved {len(chunks)} chunks to 'vectorstore/'")
