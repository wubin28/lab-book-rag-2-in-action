from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Step 1: Build a simple vector store for document retrieval
texts = [
    "RAG 1.0 used static retrieval for each query.",
    "RAG 2.0 enables feedback loops and memory integration.",
    "LangChain supports dynamic context refinement for better accuracy."
]
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)

# Step 2: Add conversational memory to persist past context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Step 3: Create a conversational retrieval chain with feedback capability
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4-turbo"),
    retriever=vectorstore.as_retriever(search_type="similarity"),
    memory=memory
)

# Step 4: Interact with the system
response1 = qa_chain.run("What was RAG 1.0 designed to do?")
response2 = qa_chain.run("And how does RAG 2.0 improve that?")
print(response2)