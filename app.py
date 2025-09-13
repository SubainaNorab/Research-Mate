from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
#loading db

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

#chat history
history=ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#LLM
hf_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    task="conversational",   
    huggingfacehub_api_token="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" 
)

llm = ChatHuggingFace(llm=hf_llm)

#chatbot
retriever=db.as_retriever(search_kwargs={"k":3})
chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,memory=history)

while True:
    query = input("\nYou: ")
    if query.lower() in ["quit", "exit"]:
        break
    response = chain.invoke({"question": query})
    print(f"Bot: {response}")

# frontend