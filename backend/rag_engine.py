import os
import re
import time 
from langchain_community.document_loaders import PyMuPDFLoader
import io
import fitz  # this is PyMuPDF's actual import name
import easyocr
import numpy as np
from PIL import Image
from langchain_core.documents import Document
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

# Main LLM - used for the final, important answer
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Smaller/faster LLM - used for cheap repetitive "mini-summary" work (Bug 2)
fast_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# --- Bug 4: Manga OCR setup ---

# Loading the OCR engine takes a few seconds, so we only load it ONCE,
# and only if someone actually uploads a manga (not every time the app starts).
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        print("Loading OCR engine for the first time... this may take a moment.")
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader


def extract_text_from_manga(file_path: str) -> list:
    """
    Manga/comics are basically pictures of text, so a normal PDF text-reader
    can't see anything. Instead, we turn each page into an image (like taking
    a screenshot of it), then use OCR to 'read' the dialogue out of the picture.
    Returns a list of Documents, just like PyMuPDFLoader normally would.
    """
    reader = get_ocr_reader()
    pdf = fitz.open(file_path)
    docs = []

    total_pages = len(pdf)
    
    test_limit = None
    pages_to_process = min(total_pages, test_limit) if test_limit else total_pages

    print(f"Starting OCR on {pages_to_process}/{total_pages} manga pages... this will take a while on CPU.")

    for page_number in range(pages_to_process):
        page = pdf[page_number]
        print(f"OCR processing page {page_number + 1}/{pages_to_process}...")

        # Render this PDF page as an image
        pix = page.get_pixmap(dpi=100)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        image_np = np.array(image)

        # Ask OCR to read all the text it can find in the image
        results = reader.readtext(image_np, detail=0)  # detail=0 = just give plain text
        page_text = "\n".join(results)

        if not page_text.strip():
            page_text = "[No readable text detected on this page]"

        docs.append(Document(
            page_content=page_text,
            metadata={"page": page_number, "source": file_path}
        ))

    return docs
# Keywords that suggest the user wants a WHOLE-BOOK answer, not a specific detail
GLOBAL_KEYWORDS = [
    "summarize", "summary", "entire book", "whole book", "overall",
    "whole novel", "entire novel", "main theme", "best problem",
    "all chapters", "complete book", "throughout the book", "in general"
]


def is_global_question(user_message: str) -> bool:
    msg = user_message.lower()
    return any(keyword in msg for keyword in GLOBAL_KEYWORDS)


def process_and_store_document(file_path: str, book_type: str = "coding") -> str:
    """
    Processes the uploaded file based on the book_type and stores it in ChromaDB.
    """
    if book_type == "manga":
        # Manga pages are images, so we OCR them instead of reading a text layer
        docs = extract_text_from_manga(file_path)
    else:
        # 1. Load the PDF with PyMuPDF (normal text-based books)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

    # Number every line on every page (needed for "page X line Y" questions)
    for doc in docs:
        lines = doc.page_content.split("\n")
        numbered_lines = [f"Line {i + 1}: {line}" for i, line in enumerate(lines)]
        doc.page_content = "\n".join(numbered_lines)

    # 2. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    # Stamp the page number into every chunk's actual text
    for chunk in chunks:
        page_number = chunk.metadata.get("page", 0) + 1
        chunk.page_content = f"[Source: PDF Viewer Page {page_number}]\n{chunk.page_content}"

    # Give each book_type its own isolated collection so books don't mix together
    try:
        old_db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name=book_type
        )
        old_db.delete_collection()
    except Exception:
        pass

    # 3. Store in Vector DB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=book_type
    )
    return "Success"


def query_rag_system(user_message: str, book_type: str = "coding", chat_history: list = None) -> str:
    """
    Queries the Vector DB and generates an answer using the LLM.
    """
    
    if llm is None:
        return "ERROR: You must initialize an LLM in rag_engine.py first!"

    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=book_type
    )

    page_match = re.search(r"page\s+(\d+)", user_message.lower())

    if page_match:
        # Exact page lookup via metadata filter (Bug 1)
        target_page_display = int(page_match.group(1))
        target_page_metadata = target_page_display - 1
        results = vector_db.get(where={"page": target_page_metadata})
        context = "\n\n".join(results["documents"]) if results and results["documents"] else "No content found for this page."

    elif is_global_question(user_message):
        # Map-Reduce for whole-book questions, with token-budget cap (Bug 2)
        all_data = vector_db.get()
        all_docs = all_data["documents"]

        batch_size = 10          # smaller batches = fewer tokens per single call
        max_batches = 6          # slightly fewer batches = less total work

        batches = [all_docs[i:i + batch_size] for i in range(0, len(all_docs), batch_size)]

        if len(batches) > max_batches:
            step = len(batches) / max_batches
            batches = [batches[int(i * step)] for i in range(max_batches)]

        summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the key points of the following textbook excerpt in 3-5 sentences:\n\n{text}"
        )
        summary_chain = summary_prompt | fast_llm | StrOutputParser()


        mini_summaries = []
        try:
            for batch in batches:
                batch_text = "\n\n".join(batch)
                mini_summaries.append(summary_chain.invoke({"text": batch_text}))
                time.sleep(3)  # small pause so we don't burst past the per-minute token limit
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                return ("The AI hit Groq's free-tier rate limit while reading this large book. "
                        "Try again in a couple of minutes, or ask about a smaller section instead.")
            raise

        context = "\n\n".join(mini_summaries)

    else:
        # Build a smarter search query: blend the current question with the
        # user's last question, so vague follow-ups like "tell me more about
        # that" or "what page is that from?" still find the right content.
        search_query = user_message
        if chat_history:
            previous_user_messages = [turn["content"] for turn in chat_history if turn["role"] == "user"]
            if previous_user_messages:
                search_query = f"{previous_user_messages[-1]} {user_message}"

        # Manga uses a higher k because the book is large (200+ pages) and
        # OCR text is noisier, so we need to search a wider net to find the
        # right page. Coding/Novel stay lower to protect the token budget.
        k_value = 12 if book_type == "manga" else 8

        retriever = vector_db.as_retriever(search_kwargs={"k": k_value})
        docs = retriever.invoke(search_query)
        context = "\n\n".join(doc.page_content for doc in docs)

    # Turn the chat_history list into readable text like "User: ...\nAssistant: ..."
    if chat_history:
        history_lines = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in chat_history]
        history_text = "\n".join(history_lines)
    else:
        history_text = "No previous conversation."

    template = """You are a helpful Interactive Study Tutor.

You have two sources of information:
1. Conversation History - what has been said so far in this chat.
2. Context - excerpts from the textbook. Each excerpt starts with a tag like "[Source: PDF Viewer Page X]" showing exactly where it came from.

RULES:
- If the question is about the TEXTBOOK CONTENT (facts, explanations, definitions, page numbers, etc.), answer ONLY using the Context below. If the Context does not contain the answer, say "I cannot find the answer to this in the textbook." Do not hallucinate or guess.
- Whenever you answer using the Context, always mention the page number it came from, using the "[Source: PDF Viewer Page X]" tag in that excerpt. For example: "Lists are sequences of values (Page 34)."
- If the question is about the CONVERSATION ITSELF (e.g. "what did I just ask", "where was that written", "what page was that on", "repeat that"), answer using the Conversation History instead, even if it's not in the Context — since a page number you already mentioned earlier is now part of the history.

Conversation History:
{history}

Context: {context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke({"context": context, "question": user_message, "history": history_text})
