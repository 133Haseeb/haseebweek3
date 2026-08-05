import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize Vector DB and Embeddings
DB_DIR = "./chroma_db"
os.makedirs(DB_DIR, exist_ok=True)

# Using local HuggingFace embeddings (runs on CPU for free!)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# TODO: Interns must configure their own LLM!
# You can use an API (OpenAI, Gemini, Groq) or a local model (Ollama).
llm = None  # REPLACE THIS WITH YOUR LLM INITIALIZATION

def process_and_store_document(file_path: str, book_type: str = "coding") -> str:
    """
    Processes the uploaded file based on the book_type and stores it in ChromaDB.
    """
    # INTERN CHALLENGE 5: The Manga OCR Challenge
    # PyMuPDF only reads text. If book_type == "manga", it will fail to read images!
    # How can you implement an OCR library (like pytesseract or EasyOCR) or a Vision API here?
    
    if book_type == "manga":
        pass # TODO: Implement Image-to-Text OCR logic for Manga!
        
    # 1. Load the PDF with PyMuPDF
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # 2. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    
    # INTERN CHALLENGE 1: Vector Blindness (Absolute Precision)
    # The text chunks here do not have the physical page numbers injected into their text body.
    # Therefore, the Semantic Search will fail if a user asks "What is on page 34?"
    # How can you inject `[Source: PDF Viewer Page X]` directly into `chunk.page_content`?
    
    # 3. Store in Vector DB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return "Success"

def query_rag_system(user_message: str, book_type: str = "coding") -> str:
    """
    Queries the Vector DB and generates an answer using the LLM.
    """
    if llm is None:
        return "ERROR: You must initialize an LLM in rag_engine.py first!"
        
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # INTERN CHALLENGE 2: Myopic Context Limits
    # The retriever only pulls 30 chunks. It cannot read a whole book!
    # How can you route Global questions differently so your LLM can read massive sections at once?
    retriever = vector_db.as_retriever(search_kwargs={"k": 30})
    
    # 2. Build the Prompt
    template = """You are a helpful Interactive Study Tutor. Answer the question based ONLY on the following context from the textbook. 
If the context does not contain the answer, say "I cannot find the answer to this in the textbook." Do not hallucinate or guess.

Context: {context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 3. Create the RAG Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 4. Generate the answer
    return rag_chain.invoke(user_question)
