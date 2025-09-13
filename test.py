from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Define Hugging Face endpoint
hf_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",   # ✅ required for chat models
    huggingfacehub_api_token="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"
)

# Step 2: Wrap endpoint in ChatHuggingFace
llm = ChatHuggingFace(llm=hf_llm)   # ✅ use `llm=`, not `endpoint=`

# Step 3: Create a chat prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI tutor."),
    ("human", "What is the relationship between extended reality and AI?")
])

# Step 4: Run the model
response = llm.invoke(prompt.format_messages())
print(response.content)
