import os
import json
import re
import uuid
import shutil
from typing import List, Dict, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup

# Import RAG modules
from pdf_processor import PDFProcessor
from vector_store import VectorStore

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # Custom Search Engine ID

# RAG configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)

# Initialize RAG components
pdf_processor = PDFProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
vector_store = VectorStore(persist_directory=VECTOR_DB_PATH)

class UserInput(BaseModel):
    message: str
    role: str
    conversation_id: str
    use_web_search: bool = False  # Optional flag for web search

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.active: bool = True
        self.document_ids: List[str] = []  # Track uploaded documents for this conversation

conversations: Dict[str, Conversation] = {}

# ------------------ Helper functions ------------------

def get_iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()
    return conversations[conversation_id]


# Generic Groq chat call wrapper
def groq_chat(messages: List[Dict[str, str]], *, model: str = "llama-3.1-8b-instant",
              temperature: float = 0.0, max_tokens: int = 1024, stream: bool = False) -> str:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=stream,
        )

        if stream:
            response = ""
            for chunk in completion:
                response += chunk.choices[0].delta.content or ""
            return response
        else:
            return completion.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}")


# ---------------- Classification ----------------
def classify_need_search(conversation: Conversation, user_message: str) -> Dict[str, Optional[str]]:
    """Returns {"search": bool, "reason": str} by asking Groq to classify."""
    timestamp = get_iso_timestamp()

    system_prompt = (
        "You are an assistant that decides whether a user's question requires a fresh web search.\n"
        "Return ONLY a JSON object with two keys: 'search' (true or false) and 'reason' (short text).\n"
        "Use the conversation context and the user message to decide.\n"
        "If the question asks about recent events, prices, schedules, live data, or contains relative words like 'today'/'yesterday', return search=true.\n"
        "If the question is general knowledge, conceptual, or can be answered from context, return search=false.\n"
        "Do not output any extra text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Timestamp: {timestamp}\nConversation context: {json.dumps(conversation.messages)}\nUser message: {user_message}"}
    ]

    try:
        raw = groq_chat(messages, temperature=0.0, max_tokens=200, stream=False)
        parsed = json.loads(raw)
        if "search" in parsed and "reason" in parsed:
            return {"search": bool(parsed["search"]), "reason": str(parsed["reason"])}
        else:
            return {"search": False, "reason": "classification missing keys - fallback to no search"}
    except Exception:
        # fallback heuristic
        low_confidence = {
            "search": any(k in user_message.lower() for k in ["today", "now", "current", "price", "rate", "latest", "news", "score", "schedule", "when", "who is the president", "who is the ceo"]),
            "reason": "fallback heuristic used"
        }
        return low_confidence


# ---------------- Generate optimized search query via Groq ----------------
def generate_search_query_via_groq(conversation: Conversation, user_message: str, timestamp: Optional[str] = None) -> str:
    """Ask Groq to produce a concise search query optimized for web search.
    The model should return a single-line search query string (no JSON)."""
    if timestamp is None:
        timestamp = get_iso_timestamp()

    system_prompt = (
        "You are a Google search query optimization assistant.\n"
        "Convert the user's message into the best possible search query.\n"
        "Rules:\n"
        "- Use the provided timestamp to resolve words like today, yesterday, this week, latest. Convert relative dates into YYYY-MM-DD when appropriate.\n"
        "- Remove filler and stop words.\n"
        "- Do NOT use a question format.\n"
        "- Add missing keywords like price, news, result, release, review, comparison when helpful.\n"
        "- Keep it under 15 words.\n"
        "- Return ONLY the optimized search query on a single line."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Timestamp: {timestamp}\nConversation context: {json.dumps(conversation.messages)}\nUser message: {user_message}\n\nProvide a single-line search query for use with Google Custom Search."}
    ]

    try:
        raw = groq_chat(messages, temperature=0.0, max_tokens=80, stream=False)
        query_line = raw.splitlines()[0].strip()
        if not query_line:
            return user_message
        return query_line
    except Exception:
        return user_message


# ---------------- Google Custom Search (snippets) ----------------
def google_search_snippets(query: str, num_results: int = 5, timestamp_iso: Optional[str] = None) -> List[Dict[str, str]]:
    """Run Google Custom Search. If timestamp_iso provided, append YYYY-MM-DD to bias results."""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise RuntimeError("Google search keys not set in environment")

    # Note: Date is already added by the AI model during query generation
    # No need to append it again here to avoid duplicates

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results
        # optionally add: "dateRestrict": "d7" to restrict to last 7 days
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    # Debug logging
    print(f"🔍 Google Search Debug:")
    print(f"   Query sent: {query}")
    print(f"   Total results available: {data.get('searchInformation', {}).get('totalResults', 'unknown')}")
    print(f"   Items returned: {len(data.get('items', []))}")
    
    # Check for errors or warnings in response
    if 'error' in data:
        print(f"   ⚠️ API Error: {data['error']}")
    if len(data.get('items', [])) == 0:
        print(f"   ⚠️ No results found. This could be due to:")
        print(f"      - API quota exceeded")
        print(f"      - CSE configuration (check if it's set to 'Search the entire web')")
        print(f"      - Query too specific")

    results = []
    for item in data.get("items", [])[:num_results]:
        results.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link")
        })
    return results


# ---------------- Fetch & extract page text ----------------
def extract_text_from_url(url: str, char_limit: int = 4000) -> str:
    try:
        headers = {"User-Agent": "chatbot/1.0 (+https://example.com)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        for s in soup(["script", "style", "noscript", "header", "footer", "iframe"]):
            s.decompose()

        article = soup.find("article")
        texts = []
        if article:
            for p in article.find_all("p"):
                text = p.get_text(strip=True)
                if text:
                    texts.append(text)
        else:
            body = soup.body
            if body:
                for p in body.find_all("p"):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:
                        texts.append(text)

        joined = "\n\n".join(texts)
        if not joined:
            meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if meta and meta.get("content"):
                joined = meta.get("content")

        joined = re.sub(r"\s+", " ", joined or "").strip()
        if len(joined) > char_limit:
            joined = joined[:char_limit] + "..."

        return joined
    except Exception:
        return ""


def enrich_search_results_with_extraction(snippets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    enriched = []
    for i, item in enumerate(snippets, 1):
        link = item.get("link")
        extracted = ""
        if link:
            try:
                print(f"  📄 Extracting from result {i}: {link[:60]}...")
                extracted = extract_text_from_url(link)
                if extracted:
                    print(f"    ✅ Extracted {len(extracted)} characters")
                else:
                    print(f"    ⚠️  No content extracted (empty)")
            except Exception as e:
                print(f"    ❌ Extraction failed: {e}")
                extracted = ""
        enriched.append({
            "title": item.get("title"),
            "link": link,
            "snippet": item.get("snippet"),
            "extracted_text": extracted
        })

    
    
    print(f"\n📊 Enriched Results Summary:")
    for i, result in enumerate(enriched, 1):
        print(f"  Result {i}:")
        print(f"    Title: {result['title'][:60]}...")
        print(f"    Snippet length: {len(result.get('snippet', ''))} chars")
        print(f"    Extracted length: {len(result.get('extracted_text', ''))} chars")
    
    return enriched


# ---------------- Build final messages for Groq ----------------
def build_refinement_messages(conversation: Conversation, user_message: str, *, search_results: Optional[List[Dict[str, str]]] = None, rag_context: Optional[List[Dict]] = None) -> List[Dict[str, str]]:
    timestamp = get_iso_timestamp()

    # Build intelligent system prompt
    if search_results:
        system_content = """You are a helpful assistant. Answer the user's question using the web search results provided below. Be direct and informative."""
    elif rag_context:
        system_content = """You are a helpful assistant. Answer the user's question using the document content provided below."""
    else:
        system_content = """You are a helpful assistant. Answer the user's question using your knowledge. If you don't know something, say so."""
    
    system_msg = {
        "role": "system",
        "content": system_content
    }

    # Simplified context message (removed verbose conversation dump for speed)
    context_msg = {
        "role": "user", 
        "content": f"Current time: {timestamp}\nUser question: {user_message}"
    }

    # Build messages with full conversation history for context awareness
    messages = [system_msg]
    
    # Add conversation history (excluding the system message we already added)
    for msg in conversation.messages[1:]:  # Skip first system message
        messages.append(msg)
    
    # Add current context
    messages.append(context_msg)
    
    # Add RAG context if available (simplified format)
    if rag_context:
        rag_text = "Uploaded Document Content:\n"
        for i, ctx in enumerate(rag_context, 1):
            rag_text += f"\n[Excerpt {i}]: {ctx['chunk']}\n"
        messages.append({"role": "assistant", "content": rag_text})

    # Add web search results if available (simplified for speed)
    if search_results:
        search_text = "Web Search Results:\n"
        for i, r in enumerate(search_results[:3], 1):  # Limit to top 3 for speed
            search_text += f"\n{i}. {r.get('title', 'N/A')}\n"
            search_text += f"   Snippet: {r.get('snippet', '')[:1000]}\n"
            
            # Append extracted_text if available
            extracted_text = r.get('extracted_text', '')
            if extracted_text:
                search_text += f"   Extracted Content: {extracted_text[:2000]}\n"
        
        messages.append({"role": "system", "content": search_text})
        
        # Print the combined search_text with both snippet and extracted_text
        print("✅ Combined search text results (snippet + extracted_text):")
        print(search_text)

    # Simple instruction for response
    messages.append({"role": "user", "content": "Answer the question based on the information provided."})
    return messages


# ---------------- Endpoint ----------------
@app.post("/chat")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(status_code=400, detail="The chat session has ended. Please start a new session.")

    # Append the user's message to the conversation
    conversation.messages.append({
        "role": input.role,
        "content": input.message
    })

    # Check if there are uploaded documents for RAG
    rag_context = None
    if conversation.document_ids:
        try:
            # Create query embedding
            query_embedding = pdf_processor.create_query_embedding(input.message)
            
            # Use hybrid query for better mid-page content retrieval
            rag_results = vector_store.hybrid_query(
                query_embedding=query_embedding,
                query_text=input.message,
                top_k=5
            )
            
            # Check relevance: only use RAG if top result is actually relevant
            # Cosine distance: 0 = identical, 2 = opposite
            # If distance > 0.6, the question is probably not about the document
            if rag_results and len(rag_results) > 0:
                top_distance = rag_results[0].get('distance', 1.0)
                if top_distance <= 0.6:  # Only use if reasonably relevant
                    rag_context = rag_results
                else:
                    # Question is generic, don't use document context
                    print(f"RAG context not relevant (distance: {top_distance:.2f}), using general knowledge")
                    rag_context = None
            
        except Exception as e:
            # Continue without RAG if it fails
            print(f"RAG query failed: {e}")
            rag_context = None
    
    # Perform web search only if explicitly requested by user
    search_results = None
    if input.use_web_search:
        print(f" Web search triggered for query: {input.message}")
        try:
            timestamp = get_iso_timestamp()
            
            # Generate optimized search query
            search_query = generate_search_query_via_groq(conversation, input.message, timestamp)
            print(f" Generated search query: {search_query}")
            
            # Perform Google Custom Search
            print(f" Calling Google Custom Search API...")
            snippets = google_search_snippets(search_query, num_results=5, timestamp_iso=timestamp)
            print(f"Got {len(snippets)} search results")
            
            # Enrich with page extraction
            print(f" Extracting page content...")
            search_results = enrich_search_results_with_extraction(snippets)
            print(f" Web search completed successfully")
        except Exception as e:
            # Continue without search if it fails
            print(f" Web search failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            search_results = None
    
    # Build messages for Groq with RAG context and/or search results
    try:
        messages = build_refinement_messages(
            conversation, 
            input.message, 
            search_results=search_results,
            rag_context=rag_context
        )
        
        # Generate response using Groq (optimized for speed and accuracy)
        response = groq_chat(messages, temperature=0.5, max_tokens=1024, stream=False)
        
        # Append assistant response to conversation
        conversation.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Return response with metadata
        return {
            "response": response,
            "used_rag": rag_context is not None and len(rag_context) > 0,
            "used_search": search_results is not None and len(search_results) > 0
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


# ---------------- RAG Endpoints ----------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), conversation_id: str = "default"):
    """
    Upload and process a PDF file for RAG.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Check file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400, 
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB"
            )
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save file to disk
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Process PDF
        try:
            result = pdf_processor.process_pdf(file_path)
            
            # Store in vector database
            metadata = {
                "filename": file.filename,
                "upload_date": get_iso_timestamp(),
                "conversation_id": conversation_id,
                "file_size_mb": round(file_size_mb, 2)
            }
            
            vector_store.add_document(
                doc_id=doc_id,
                chunks=result["chunks"],
                embeddings=result["embeddings"],
                metadata=metadata
            )
            
            # Add document to conversation
            conversation = get_or_create_conversation(conversation_id)
            if doc_id not in conversation.document_ids:
                conversation.document_ids.append(doc_id)
            
            return {
                "success": True,
                "doc_id": doc_id,
                "filename": file.filename,
                "num_chunks": result["num_chunks"],
                "message": f"Successfully processed {file.filename}"
            }
        
        except Exception as e:
            # Clean up file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents(conversation_id: Optional[str] = None):
    """
    List all uploaded documents, optionally filtered by conversation.
    """
    try:
        all_docs = vector_store.list_documents()
        
        # Filter by conversation if specified
        if conversation_id:
            conversation = conversations.get(conversation_id)
            if conversation:
                all_docs = [
                    doc for doc in all_docs 
                    if doc["doc_id"] in conversation.document_ids
                ]
        
        return {
            "documents": all_docs,
            "count": len(all_docs)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, conversation_id: Optional[str] = None):
    """
    Delete a document from the vector store and file system.
    """
    try:
        # Delete from vector store
        success = vector_store.delete_document(doc_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file from disk
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove from conversation
        if conversation_id and conversation_id in conversations:
            conversation = conversations[conversation_id]
            if doc_id in conversation.document_ids:
                conversation.document_ids.remove(doc_id)
        
        return {
            "success": True,
            "message": "Document deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def cleanup_on_shutdown():
    """
    Clean up all uploaded documents and vector store when server shuts down.
    """
    try:
        print("Server shutting down - cleaning up documents...")
        
        # Delete all documents from vector store
        all_docs = vector_store.list_documents()
        for doc in all_docs:
            try:
                vector_store.delete_document(doc["doc_id"])
            except Exception as e:
                print(f"Error deleting document {doc['doc_id']}: {e}")
        
        # Clear upload directory
        if os.path.exists(UPLOAD_DIR):
            try:
                shutil.rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR, exist_ok=True)
            except Exception as e:
                print(f"Error clearing upload directory: {e}")
        
        # Clear vector database
        if os.path.exists(VECTOR_DB_PATH):
            try:
                shutil.rmtree(VECTOR_DB_PATH)
            except Exception as e:
                print(f"Error clearing vector database: {e}")
        
        # Clear conversations
        conversations.clear()
        
        print("Cleanup completed successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
