from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


from langchain.llms import Ollama

def run_rag(query:str)->str:

    # Step 1: Load local knowledge base
    loader = TextLoader("data/knowledge.txt")  # Replace with your own file
    documents = loader.load()

    # Step 2: Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Step 3: Use local embedding model (MiniLM - free & fast)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)

    # Step 4: Use local LLM via Ollama (e.g., mistral, llama3, gemma)
    llm = Ollama(model="mistral")  # You can change to "llama3", "gemma", etc.

    # Step 5: Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    return qa_chain.run(query)