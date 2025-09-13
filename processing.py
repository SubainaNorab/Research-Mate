from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
import tempfile


def processing(uploaded_files, persist_dir="chroma_db"):

    all_docs=[]
    for file in uploaded_files:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())   # write contents to disk
            tmp_path = tmp_file.name

        # Load PDF with PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

        # Remove temp file
        os.remove(tmp_path)



    #splitting

    text_split=CharacterTextSplitter(chunk_size=500,chunk_overlap=50)

    doc_split=text_split.split_documents(all_docs)

    #embedding 
    embed_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # +storing in db
    db=Chroma.from_documents(doc_split, embedding=embed_model, persist_directory=persist_dir)

    return db

