from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

all_docs=[]
for file in os.listdir("data"):
    if file.endswith('.pdf'):
        loads=PyPDFLoader(os.path.join("data",file))
        document=loads.load()
        all_docs.extend(document)


#splitting

text_split=CharacterTextSplitter(chunk_size=500,chunk_overlap=50)

doc_split=text_split.split_documents(all_docs)

#embedding 
embed_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# +storing in db
db=Chroma.from_documents(doc_split, embed_model, persist_directory="chroma_db")

db.persist()  # save to disk

print("Stored all chunks in Chroma!")

