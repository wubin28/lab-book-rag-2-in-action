import os
import getpass
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI

# Get DeepSeek API key from user input (hidden)
print("Please enter your DeepSeek API key:")
api_key = getpass.getpass("API Key: ")

# Set environment variable for embeddings
os.environ["OPENAI_API_KEY"] = api_key

# Step 1: Create a set of up-to-date internal documents
docs = [
    "Cloud security policy updated on June 2025: All sensitive data must use AES-256 encryption.",
    "Developers must rotate API keys every 90 days according to the 2025 compliance rule.",
    "Multi-factor authentication is mandatory for all production system access."
]

# Step 2: Embed and index the documents into a vector database
# Using local HuggingFace embeddings (no external API calls needed for embeddings)
print("Loading local embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Building vector store...")
vector_db = FAISS.from_texts(docs, embeddings)
print("Vector store ready!")

# Step 3: Build a retrieval-augmented QA chain
# Using DeepSeek-reasoner model
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        model="deepseek-reasoner",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.7
    ),
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_type="similarity")
)

# Step 4: Query the assistant with a factual question
print("\nProcessing your question...")
response = qa_chain.run("What are the latest cloud security requirements?")
print("\n=== Answer ===")
print(response)