from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
import time
from processing import processing  



embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#upload file
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
#loading db

if uploaded_files:
    db = processing(uploaded_files)

    #chat history
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#LLM
    hf_llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
        task="conversational",   
        huggingfacehub_api_token="" 
    )

    llm = ChatHuggingFace(llm=hf_llm)

    #chatbot
    retriever=db.as_retriever(search_kwargs={"k":3})
    chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,memory=st.session_state.memory)

    # frontend

    with st.sidebar:
        st.title("Research Mate")
        st.markdown("""
        **Your AI Research Assistant**  
        - Upload AI-related PDFs  
        - Ask research questions  
        - Get concise, reliable answers  
        
        Powered by LangChain + HuggingFace
        """)
        st.divider()
    st.title("Research Mate")
    st.subheader("AI Research Mate ChatBot")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask a research question..."):
        # Show user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Placeholder for assistant reply
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ Thinking...")  # small loader only

            # Simulate response delay (replace with chain call)
            time.sleep(2)
            response = chain.invoke({"question": prompt})
            bot_reply = response["answer"]

            # bot_reply = "Here is the summary of your AI-related query..."  # example

            # Replace loader with actual answer
            placeholder.markdown(bot_reply)

        # Save assistant response in history
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

else:
    st.info("Upload PDFs First")