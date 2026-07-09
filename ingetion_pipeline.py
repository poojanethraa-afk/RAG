import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langhain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

def main():
    print("main")
    documents= load_documents(docs_path="docs")
    chunks = split_documents(documents)
    vectorstore= create_vector_store(chunks)

def load_documents(docs_path="docs"):
    
    if not os.path.exists(docs_path):
        raise ValueError(f"Directory {docs_path} does not exist."
    loader= DirectoryLoader(path=docs_path, glob= "*.txt", loader_cls=TextLoader)

    documents = loader.load() 

if len(documents) == 0:
    raise FileNotFoundError(f"No documents found in {docs_path}. Please add some .txt files to the directory.")

return documents

def split_documents(documents, chunk_size=800, chunk_overlap= 0):
    text_splitter= CharaterTextSplitter(chunks_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks= text_splitter.split_documents(documents)
    return chunks
 def create_vector_store(chunks, persist_directory= "db/chroma_db"):
 
 embedding_model= OpenAIEmbeddings(model= "text-embedding-3-small")

 vector_score= Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"})
  

 
   

