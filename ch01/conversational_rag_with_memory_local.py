import getpass
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# Prompt user for DeepSeek API key securely (input will not be displayed)
print("Please enter your DeepSeek API key:")
deepseek_api_key = getpass.getpass("")

# Step 1: Build a simple vector store for document retrieval
texts = [
    "RAG 1.0 used static retrieval for each query.",
    "RAG 2.0 enables feedback loops and memory integration.",
    "LangChain supports dynamic context refinement for better accuracy."
]

# Use local HuggingFace embeddings (no external API calls needed)
print("Loading local embedding model (first time may take a moment)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Building vector store...")
vectorstore = FAISS.from_texts(texts, embeddings)
print("Vector store ready!")

# Step 2: Add conversational memory to persist past context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Step 3: Create a conversational retrieval chain with feedback capability
# Using DeepSeek-reasoner model
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
        model="deepseek-reasoner",
        openai_api_key=deepseek_api_key,
        openai_api_base="https://api.deepseek.com/v1"
    ),
    retriever=vectorstore.as_retriever(search_type="similarity"),
    memory=memory
)

# Step 4: Interact with the system
print("\nQuerying the conversational RAG system...")
print("\nQuery 1: What was RAG 1.0 designed to do?")
response1 = qa_chain.run("What was RAG 1.0 designed to do?")
print(f"Response 1: {response1}\n")

print("Query 2: And how does RAG 2.0 improve that?")
response2 = qa_chain.run("And how does RAG 2.0 improve that?")
print(f"Response 2: {response2}")

