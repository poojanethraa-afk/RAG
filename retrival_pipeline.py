from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
persistent_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db= Chroma(persist_directory= persistent,
           embedding_function= embedding_model,
           collection_name= {"hnsw:space": "cosine"}
query = "Which island does spacx lease for its launches in the specific?"
retrever= db.as_retriever(search_kwargs={"k": 3})

relevent_docs= retrever.invoke(query)
