# export OPENAI_API_KEY=sk-123456789abcdef
# pip install -e .
# python3 src/api/index.py
# uvicorn src.api.index:app --reload
# http://localhost:8000/docs
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import os
import time
import logging
import httpx

# =====================================================================
# Logging setup
# =====================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pocketagent.chatwoot")

# ----- Clean imports (no sys.path needed)
from pocketagent.pocketagent_app import PocketAgentApp
from pocketagent.ragagent.ragagent import RagAgent
from pocketagent.ragagent.ragbot import RAGBot
from pocketagent.mcpagent.mcpagent import MCPAgent
# ----- LLM / Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

# =====================================================================
# Config
# =====================================================================
WORKING_DIR = os.getcwd()
MCP_SERVER_PATH = os.path.join(WORKING_DIR, "src", "mcpserver", "server.py")


import os
WORKING_DIR = os.getcwd()
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MD_PATH_STR = os.path.join(WORKING_DIR, "product_faq.md")
FAISS_PATH = os.path.join(WORKING_DIR, "faiss_index")
MCP_SERVER_COMMAND = "python3"
MCP_SERVER_PATH = os.path.join(WORKING_DIR, "src", "mcpserver", "server.py")
MCP_SERVER_ARGS = [MCP_SERVER_PATH]
# =====================================================================
# Core singletons (safe to build synchronously)
# =====================================================================
llm = ChatOpenAI(model="gpt-4o-mini")
emb = OpenAIEmbeddings(model="text-embedding-3-small")
memory_saver = MemorySaver()  # Global memory for persistence

# =====================================================================
# Lifespan context manager
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    #-------------------------------------
    # RagAgent setup
    rag_agent = RagAgent(RAGBot(
        llm=llm,
        emb=emb,
        md_path_str=MD_PATH_STR,
        faiss_path=FAISS_PATH
    ))

    #-------------------------------------
    # MCPAgent setup
    # Using langchain_mcp_adapters to create a MultiServerMCPClient
    from langchain_mcp_adapters.client import MultiServerMCPClient
    mcp_client = MultiServerMCPClient(
        {
            "math": {
                "command": MCP_SERVER_COMMAND,
                "args": MCP_SERVER_ARGS,
                "transport": "stdio",
            }
        }
    )
    tools = await mcp_client.get_tools()
    mcp_agent = MCPAgent(llm=llm,tools=tools)

    #-------------------------------------
    # Create PocketAgentApp with persistent memory
    pocket_app = PocketAgentApp(llm=llm, emb=emb, rag_agent=rag_agent, mcp_agent=mcp_agent, memory_saver=memory_saver)

    app.state.pocket_app = pocket_app
    app.state.mcp_client = mcp_client

    yield

    # Cleanup
    await mcp_client.close()

# =====================================================================
# FastAPI app
# =====================================================================
app = FastAPI(title="PocketAgent API", lifespan=lifespan)

class ChatRequest(BaseModel):
    text: str
    thread_id: Optional[str] = "demo"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

# Encapsulated chat logic
async def run_pocketagent_chat(text: str, thread_id: str) -> str:
    from langchain_core.messages import HumanMessage
    config = {"configurable": {"thread_id": thread_id}}

    try:
        out = await app.state.pocket_app.get_graph().ainvoke({
            "messages": [HumanMessage(content=text)],
            "intent": "",
            "thread_id": thread_id
        }, config)
        return out["messages"][-1].content
    except Exception as e:
        logger.error("Error in chat: %s", e)
        return f"Sorry, I encountered an error: {str(e)}"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    response = await run_pocketagent_chat(req.text, req.thread_id)
    return ChatResponse(response=response, thread_id=req.thread_id)

@app.get("/ping")
def ping():
    return {"ok": True, "status": "PocketAgent API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "components": {
            "llm": "gpt-4o-mini",
            "embeddings": "text-embedding-3-small",
            "rag_enabled": True,
            "mcp_enabled": True
        }
    }

# =====================================================================
# Chatwoot webhook callback
# =====================================================================
from fastapi.responses import HTMLResponse
from pocketagent.chatwoot.shopify_router import classify_email
from pocketagent.chatwoot.actions import notify_discord, send_return_link, resolve_conversation
from pocketagent.chatwoot.logstore import log_store

COOLDOWN_SECONDS = 30
_last_bot_reply: dict[int, float] = {}

def _is_on_cooldown(conversation_id: int) -> bool:
    last = _last_bot_reply.get(conversation_id)
    return bool(last and (time.time() - last) < COOLDOWN_SECONDS)

def _mark_replied(conversation_id: int):
    _last_bot_reply[conversation_id] = time.time()

async def _last_message_is_outgoing(conversation_id: int, headers: dict) -> bool:
    """Check via Chatwoot API if the last message in the conversation is already from the bot/agent."""
    try:
        chatwoot_host = os.environ.get('CHATWOOT_HOST', 'localhost:3000')
        chatwoot_account = os.environ.get('CHATWOOT_ACCOUNT_ID', '2')
        url = f"http://{chatwoot_host}/api/v1/accounts/{chatwoot_account}/conversations/{conversation_id}/messages"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                messages = resp.json().get("payload", [])
                if messages and messages[0].get("message_type") in (1, "outgoing"):
                    return True
    except Exception as e:
        logger.error("Error checking last message: %s", e)
    return False

def _get_agent_base_url() -> str:
    """Return the external base URL for the agent (for log links in messages)."""
    return os.environ.get("AGENT_BASE_URL", "http://localhost:8000")

@app.post("/chatwootcb")
async def chatwootcb(request: Request):
    body = await request.json()
    logger.debug("Webhook payload: %s", body)

    # Decision trace: collect which nodes were visited and their outcome
    trace = []

    # Guardrail 1: Only react to incoming messages (from contacts)
    message_type = body.get("message_type")
    event = body.get("event")
    trace.append(f"guardrail_incoming: message_type={message_type}")
    if message_type != "incoming":
        trace.append("RESULT: dropped (not incoming)")
        logger.info("TRACE conv=? | %s", " -> ".join(trace))
        return JSONResponse(status_code=200, content={"status": "ignored", "reason": "not an incoming message"})

    # Filter: Only process email channel messages
    conversation = body.get("conversation", {})
    channel = conversation.get("channel", "")
    inbox = body.get("inbox", {})
    inbox_channel = inbox.get("channel_type", "") if isinstance(inbox, dict) else ""
    is_email = "email" in channel.lower() or "email" in inbox_channel.lower()
    trace.append(f"filter_email_channel: channel={channel} inbox={inbox_channel} is_email={is_email}")
    if not is_email:
        trace.append("RESULT: dropped (not email)")
        logger.info("TRACE conv=? | %s", " -> ".join(trace))
        return JSONResponse(status_code=200, content={"status": "ignored", "reason": "not an email channel"})

    # Filter: Only process messages from human senders (contacts), not agents/bots
    sender = body.get("sender", {})
    sender_type = sender.get("type", "")
    sender_email = sender.get("email", "")
    trace.append(f"filter_human_sender: sender_type={sender_type}")
    if sender_type and sender_type.lower() != "contact":
        trace.append("RESULT: dropped (not contact)")
        logger.info("TRACE conv=? | %s", " -> ".join(trace))
        return JSONResponse(status_code=200, content={"status": "ignored", "reason": "sender is not a contact"})

    conversation_id = conversation.get("id")
    content = body.get("content")

    if not content or not isinstance(content, str) or not conversation_id:
        trace.append("RESULT: dropped (no content)")
        logger.info("TRACE conv=%s | %s", conversation_id, " -> ".join(trace))
        return JSONResponse(status_code=200, content={"status": "ignored", "reason": "no actionable content"})

    # Create log record
    record = log_store.create_record(
        conversation_id=conversation_id,
        sender_email=sender_email,
        content_preview=content,
    )

    trace.append(f"content_length={len(content)}")
    logger.debug("Received email in conversation %s: %s", conversation_id, content[:200])

    headers = {
        'api_access_token': os.environ.get('CHATWOOT_API_ACCESS_TOKEN', ''),
        'Content-Type': 'application/json'
    }

    # Guardrail 2: Cooldown
    on_cooldown = _is_on_cooldown(conversation_id)
    trace.append(f"guardrail_cooldown: {on_cooldown}")
    if on_cooldown:
        trace.append("RESULT: dropped (cooldown)")
        record.trace = list(trace)
        record.action = "dropped"
        record.action_detail = "cooldown active"
        log_store.save(record)
        logger.info("TRACE conv=%s | %s", conversation_id, " -> ".join(trace))
        return JSONResponse(status_code=200, content={"status": "ignored", "reason": "cooldown active"})

    # Guardrail 3: Check if last message is already from the bot
    is_outgoing = await _last_message_is_outgoing(conversation_id, headers)
    trace.append(f"guardrail_outgoing: {is_outgoing}")
    if is_outgoing:
        trace.append("RESULT: dropped (already outgoing)")
        record.trace = list(trace)
        record.action = "dropped"
        record.action_detail = "last message already outgoing"
        log_store.save(record)
        logger.info("TRACE conv=%s | %s", conversation_id, " -> ".join(trace))
        return JSONResponse(status_code=200, content={"status": "ignored", "reason": "last message already from bot"})

    # Classify the email using the Shopify router
    intent = classify_email(llm, content)
    trace.append(f"shopify_router: intent={intent}")
    record.intent = intent

    chatwoot_host = os.environ.get('CHATWOOT_HOST', 'localhost:3000')
    chatwoot_account = os.environ.get('CHATWOOT_ACCOUNT_ID', '2')
    conversation_url = f"http://{chatwoot_host}/app/accounts/{chatwoot_account}/conversations/{conversation_id}"
    log_url = f"{_get_agent_base_url()}/langlogs/{record.id}"

    if intent == "spam":
        await resolve_conversation(conversation_id, headers)
        trace.append("action: resolve_conversation (spam)")
        record.action = "resolve_conversation"
        record.action_detail = "Conversation als resolved markiert (Spam)"
    elif intent == "frustrated":
        await notify_discord(conversation_id, content, conversation_url, log_url=log_url)
        _mark_replied(conversation_id)
        trace.append("action: notify_discord (frustrated)")
        record.action = "notify_discord"
        record.action_detail = "Discord Notification gesendet"
    elif intent == "return_request":
        await send_return_link(conversation_id, headers, log_url=log_url)
        _mark_replied(conversation_id)
        trace.append("action: send_return_link")
        record.action = "send_return_link"
        record.action_detail = "Self-Service Retouren-Link gesendet"
    elif intent == "refund_status":
        await notify_discord(conversation_id, content, conversation_url, log_url=log_url)
        _mark_replied(conversation_id)
        trace.append("action: notify_discord (refund_status)")
        record.action = "notify_discord"
        record.action_detail = "Discord Notification: Kunde fragt nach Erstattungsstatus"
    else:
        trace.append("action: none (ignore)")
        record.action = "none"
        record.action_detail = "Keine Aktion (ignore)"

    record.trace = list(trace)
    log_store.save(record)
    logger.info("TRACE conv=%s | %s", conversation_id, " -> ".join(trace))
    return JSONResponse(status_code=200, content={"status": "ok", "intent": intent})

# =====================================================================
# Logging UI
# =====================================================================
from datetime import datetime, timezone

LANGLOGS_LIST_HTML = """\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>LangGraph Logs</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f5f5f5; }
  h1 { color: #333; }
  .info { color: #666; margin-bottom: 1rem; }
  table { border-collapse: collapse; width: 100%%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th, td { padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #eee; }
  th { background: #333; color: #fff; }
  tr:hover { background: #f0f7ff; }
  a { color: #0066cc; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; color: #fff; }
  .badge-spam { background: #e74c3c; }
  .badge-frustrated { background: #e67e22; }
  .badge-return_request { background: #27ae60; }
  .badge-ignore { background: #95a5a6; }
  .badge-refund_status { background: #3498db; }
  .badge-dropped { background: #7f8c8d; }
  .badge-none { background: #bdc3c7; }
  .empty { text-align: center; padding: 3rem; color: #999; }
</style>
</head>
<body>
<h1>LangGraph Logs</h1>
<p class="info">Zeigt alle Chatwoot Callback Logs der letzten 24 Stunden. %(count)d Eintraege.</p>
<table>
<thead><tr><th>Zeit</th><th>Conv ID</th><th>Absender</th><th>Intent</th><th>Aktion</th><th>Detail</th></tr></thead>
<tbody>
%(rows)s
</tbody>
</table>
</body>
</html>
"""

LANGLOGS_DETAIL_HTML = """\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Log Detail %(record_id)s</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f5f5f5; }
  h1 { color: #333; }
  a { color: #0066cc; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .card { background: #fff; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .card h2 { margin-top: 0; color: #555; font-size: 1.1em; border-bottom: 1px solid #eee; padding-bottom: 0.5rem; }
  .meta-grid { display: grid; grid-template-columns: 150px 1fr; gap: 0.3rem 1rem; }
  .meta-grid dt { font-weight: 600; color: #555; }
  .meta-grid dd { margin: 0; }
  .content-box { background: #f9f9f9; border: 1px solid #eee; border-radius: 4px; padding: 1rem; white-space: pre-wrap; word-break: break-word; font-size: 0.95em; }
  .trace-list { list-style: none; padding: 0; }
  .trace-list li { padding: 0.4rem 0.8rem; margin: 0.3rem 0; background: #f0f7ff; border-left: 3px solid #0066cc; border-radius: 0 4px 4px 0; font-family: monospace; font-size: 0.9em; }
  .trace-list li.action-node { border-left-color: #27ae60; background: #f0fff4; }
  .trace-list li.dropped-node { border-left-color: #e74c3c; background: #fff5f5; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; color: #fff; }
  .badge-spam { background: #e74c3c; }
  .badge-frustrated { background: #e67e22; }
  .badge-return_request { background: #27ae60; }
  .badge-ignore { background: #95a5a6; }
  .badge-refund_status { background: #3498db; }
  .badge-dropped { background: #7f8c8d; }
  .badge-none { background: #bdc3c7; }
</style>
</head>
<body>
<p><a href="/langlogs">&larr; Zurueck zur Uebersicht</a></p>
<h1>Log Record %(record_id)s</h1>

<div class="card">
  <h2>Meta</h2>
  <dl class="meta-grid">
    <dt>Timestamp</dt><dd>%(timestamp)s</dd>
    <dt>Conversation ID</dt><dd>%(conversation_id)s</dd>
    <dt>Absender</dt><dd>%(sender_email)s</dd>
    <dt>Intent</dt><dd><span class="badge badge-%(intent_class)s">%(intent)s</span></dd>
    <dt>Aktion</dt><dd>%(action)s</dd>
    <dt>Detail</dt><dd>%(action_detail)s</dd>
  </dl>
</div>

<div class="card">
  <h2>Eingehende Nachricht</h2>
  <div class="content-box">%(content_preview)s</div>
</div>

<div class="card">
  <h2>Entscheidungspfad (LangGraph Trace)</h2>
  <ol class="trace-list">
%(trace_items)s
  </ol>
</div>

</body>
</html>
"""


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _intent_class(intent: str | None) -> str:
    if intent in ("spam", "frustrated", "return_request", "refund_status", "ignore"):
        return intent
    return "none"


@app.get("/langlogs", response_class=HTMLResponse)
async def langlogs_list():
    records = log_store.get_all()
    if not records:
        rows = '<tr><td colspan="6" class="empty">Keine Logs vorhanden</td></tr>'
    else:
        row_list = []
        for r in records:
            ts = datetime.fromtimestamp(r.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            ic = _intent_class(r.intent)
            intent_label = r.intent or r.action or "-"
            action_label = r.action or "-"
            detail = _escape_html(r.action_detail or "-")
            row_list.append(
                f'<tr>'
                f'<td><a href="/langlogs/{r.id}">{ts}</a></td>'
                f'<td>{r.conversation_id or "-"}</td>'
                f'<td>{_escape_html(r.sender_email or "-")}</td>'
                f'<td><span class="badge badge-{ic}">{_escape_html(intent_label)}</span></td>'
                f'<td>{_escape_html(action_label)}</td>'
                f'<td>{detail}</td>'
                f'</tr>'
            )
        rows = "\n".join(row_list)
    html = LANGLOGS_LIST_HTML % {"rows": rows, "count": len(records)}
    return HTMLResponse(content=html)


@app.get("/langlogs/{record_id}", response_class=HTMLResponse)
async def langlogs_detail(record_id: str):
    record = log_store.get(record_id)
    if not record:
        return HTMLResponse(content="<h1>404 - Log Record nicht gefunden</h1>", status_code=404)

    ts = datetime.fromtimestamp(record.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ic = _intent_class(record.intent)

    trace_items = []
    for step in record.trace:
        css = ""
        if step.startswith("action:"):
            css = ' class="action-node"'
        elif "RESULT: dropped" in step:
            css = ' class="dropped-node"'
        trace_items.append(f"    <li{css}>{_escape_html(step)}</li>")

    html = LANGLOGS_DETAIL_HTML % {
        "record_id": _escape_html(record.id),
        "timestamp": ts,
        "conversation_id": record.conversation_id or "-",
        "sender_email": _escape_html(record.sender_email or "-"),
        "intent": _escape_html(record.intent or "-"),
        "intent_class": ic,
        "action": _escape_html(record.action or "-"),
        "action_detail": _escape_html(record.action_detail or "-"),
        "content_preview": _escape_html(record.content_preview),
        "trace_items": "\n".join(trace_items) if trace_items else '    <li>Kein Trace vorhanden</li>',
    }
    return HTMLResponse(content=html)


# =====================================================================
# Uvicorn entry (optional, for local runs)
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.index:app", host="0.0.0.0", port=8000, reload=True)
