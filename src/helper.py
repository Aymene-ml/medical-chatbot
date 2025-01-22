from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

#Extract Data from PDF
def load_pdf(data):
    loader = DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    return loader.load()

#Split Data into Documents
def split_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    return text_splitter.split_documents(data)

#Download Embeddings from HuggingFace

def download_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
