import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize Vector DB and Embeddings
DB_DIR = "./chroma_db"
os.makedirs(DB_DIR, exist_ok=True)

# Using local HuggingFace embeddings (runs on CPU for free!)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def process_and_store_document(file_path: str, book_type: str = "coding") -> str:
    """
    Processes the uploaded file based on the book_type and stores it in ChromaDB.
    """
    if book_type == "manga":
        pass  # We'll fix this in Bug 4

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

    # --- BUG 1 FIX: Stamp the page number into every chunk's actual text ---
    # PyMuPDFLoader stores the page number in chunk.metadata["page"], starting at 0.
    # We add 1 so it matches what a human sees in a PDF viewer (page 1, not page 0).
    for chunk in chunks:
        page_number = chunk.metadata.get("page", 0) + 1
        chunk.page_content = f"[Source: PDF Viewer Page {page_number}]\n{chunk.page_content}"
    # -------------------------------------------------------------------------

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

    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )



    # INTERN CHALLENGE 2: Myopic Context Limits
    # The retriever only pulls 30 chunks. It cannot read a whole book!
    # How can you route Global questions differently so your LLM can read massive sections at once?

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 30}
    )

    # Build the Prompt
    template = """You are a helpful Interactive Study Tutor. Answer the question based ONLY on the following context from the textbook.
If the context does not contain the answer, say "I cannot find the answer to this in the textbook." Do not hallucinate or guess.

Context: {context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG Chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Generate answer
    return rag_chain.invoke(user_message)
