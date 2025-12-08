import os
import json
import re
from typing import List, Dict, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # Custom Search Engine ID

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)

class UserInput(BaseModel):
    message: str
    role: str
    conversation_id: str

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.active: bool = True

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

    # append date to query if available
    if timestamp_iso:
        try:
            date_part = timestamp_iso.split("T", 1)[0]
            query = f"{query} {date_part}"
        except Exception:
            pass

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
    for item in snippets:
        link = item.get("link")
        extracted = ""
        if link:
            try:
                extracted = extract_text_from_url(link)
            except Exception:
                extracted = ""
        enriched.append({
            "title": item.get("title"),
            "link": link,
            "snippet": item.get("snippet"),
            "extracted_text": extracted
        })
    return enriched


# ---------------- Build final messages for Groq ----------------
def build_refinement_messages(conversation: Conversation, user_message: str, *, search_results: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    timestamp = get_iso_timestamp()

    system_msg = {
        "role": "system",
        "content": "You are a highly intelligent, friendly, and conversational assistant.Use the conversation naturally and answer like a human expert, not like a report generator.If web search results are provided, use them silently to improve accuracy.Only mention sources if the user explicitly asks for them. If some data could not be fetched, do NOT apologize or explain system limitations.Instead, answer with the best available information in a confident, clear, natural tone. If exact data is unavailable, approximate using the most recent or closest reliable information without mentioning system failure or limitations."
    }

    context_msg = {"role": "user", "content": f"Timestamp: {timestamp}\nConversation context: {json.dumps(conversation.messages)}\nUser message: {user_message}"}

    messages = [system_msg, context_msg]

    if search_results:
        trimmed_results = []
        for r in search_results:
            trimmed_results.append({
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": (r.get("snippet") or "")[:300],
                "extracted_text": (r.get("extracted_text") or "")[:2000]
            })
        results_text = json.dumps(trimmed_results, ensure_ascii=False)
        messages.append({"role": "assistant", "content": f"Search results with extracted text: {results_text}"})

    messages.append({"role": "user", "content": "Please provide a clear answer using the context and the search results above (if any). If you used extracted page text, indicate which result (title or link) you used for key facts and include brief citations."})

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

    # STEP A: Classify whether a web search is required
    classification = classify_need_search(conversation, input.message)

    try:
        if classification.get("search"):
            # STEP B: generate optimized search query using Groq
            timestamp = get_iso_timestamp()
            optimized_query = generate_search_query_via_groq(conversation, input.message, timestamp=timestamp)

            # STEP C: use optimized query to call Google Custom Search
            try:
                snippets = google_search_snippets(optimized_query, num_results=5, timestamp_iso=timestamp)
            except Exception:
                snippets = []

            # STEP D: fetch pages and extract text
            enriched = enrich_search_results_with_extraction(snippets)

            # STEP E: send enriched results + context to Groq for refined answer
            messages_for_groq = build_refinement_messages(conversation, input.message, search_results=enriched)
            final_answer = groq_chat(messages_for_groq, temperature=0.7, max_tokens=1000, stream=False)

            conversation.messages.append({"role": "assistant", "content": final_answer})

            return {
                "response": final_answer,
                "conversation_id": input.conversation_id,
                "used_search": True,
                "classification_reason": classification.get("reason"),
                "optimized_query": optimized_query,
                "search_results_count": len(enriched)
            }
        else:
            # No web search required
            messages_for_groq = build_refinement_messages(conversation, input.message, search_results=None)
            final_answer = groq_chat(messages_for_groq, temperature=0.7, max_tokens=800, stream=False)

            conversation.messages.append({"role": "assistant", "content": final_answer})

            return {"response": final_answer, "conversation_id": input.conversation_id, "used_search": False, "classification_reason": classification.get("reason")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

