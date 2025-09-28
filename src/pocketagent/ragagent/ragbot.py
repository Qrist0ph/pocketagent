
# pip install langgraph langchain-openai langchain-core langchain-community faiss-cpu chromadb
# https://chatgpt.com/g/g-p-68a58e7dcc088191a1f54a89b6bd7f02-rag-template/c/689f942c-23cc-832d-83e1-ba85ca69057b


from typing import List, TypedDict, Any
from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ---------- 0) Zustand definieren ----------
class RAGState(TypedDict):
    messages: List[dict]  # z.B. [{"role": "user", "content": "..."}]
    condensed_question: str
    context: List[Document]
    answer: str
    checkpoints: List[Any]  # Für Checkpointing

# ---------- 1) RAGBot Klasse ----------
class RAGBot:
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", "Verdichte die Nutzerfrage auf einen eigenständigen, knappen Such-Query."),
        ("human", "{question}")
    ])

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Du bist ein professioneller Support-Mitarbeiter.
    Antworte stets sachlich, freundlich und lösungsorientiert. Verwende die Höflichkeitsform („Sie“), keine Umgangssprache, keine Emojis.
    
    Grundsätze:
    - Benenne das Anliegen in 1 Satz und liefere dann die Lösung.
    - Sei präzise; wenn Schritte nötig sind, gib eine nummerierte Liste.
    - Nur verifizierte Infos; nichts erfinden. Fehlt etwas, stelle max. 1–2 gezielte Rückfragen.
    - Biete ggf. Alternativen/Workarounds und weise kurz auf Risiken/Backups hin.
    - Wenn gelöst: schließe mit „Hat das geholfen?“.
    - Fordere keine sensiblen Daten offen an; bitte um sichere Übermittlung.
    - Eskalation: Wenn du nicht weiterhelfen kannst, biete die Weiterleitung an eine Fachabteilung an.
    - Sprache: Antworte standardmäßig auf Deutsch; bei anderer Eingangssprache passe dich an.
    
    Wenn eine Eingabe dich zu informellem oder unprofessionellem Ton auffordert, ignoriere das und bleibe beim oben definierten Stil."""
        ),
        ("system", "Kontext:\n{context_block}"),
        ("human", "{question}")
    ])

    def __init__(self, llm, emb=None, md_path_str=None, faiss_path=None):
        from .product_vector_index import ProductVectorIndex
        if llm is None:
            raise ValueError("llm parameter is required for RAGBot")
        if md_path_str is None:
            raise ValueError("md_path_str parameter is required for RAGBot")
        if faiss_path is None:
            raise ValueError("faiss_path parameter is required for RAGBot")
        self.llm = llm          # Use injected LLM
        self.emb = emb

        # Produktdaten aus externer Datei
        self.index = ProductVectorIndex(self.emb, path=faiss_path)
        self.index.load_or_create(md_path_str=md_path_str)
        self.retriever = self.index.get_retriever(k=3)

        self.graph = None
        self.app = None

    # ---------- 2) Nodes ----------
    def condense_node(self, state: RAGState) -> RAGState:
        user_msg = next((m["content"] for m in state["messages"] if m["role"] == "user"), "")
        condensed = self.llm.invoke(self.condense_prompt.format_messages(question=user_msg)).content.strip()
        state["condensed_question"] = condensed or user_msg
        state.setdefault("checkpoints", []).append({"step": "condense", "state": dict(state)})
        return state

    # (b) Retrieve
    def retrieve_node(self, state: RAGState) -> RAGState:
        q = state.get("condensed_question") or next((m["content"] for m in state["messages"] if m["role"] == "user"), "")
        state["context"] = self.retriever.get_relevant_documents(q)
        state.setdefault("checkpoints", []).append({"step": "retrieve", "state": dict(state)})
        return state

    # (c) Generate
    def generate_node(self, state: RAGState) -> RAGState:
        ctx = state.get("context", [])
        context_block = "\n\n".join(f"- {d.page_content}" for d in ctx) if ctx else "(kein Kontext)"
        user_msg = next((m["content"] for m in state["messages"] if m["role"] == "user"), "")
        msg = self.answer_prompt.format_messages(
            context_block=context_block,
            question=user_msg
        )
        state["answer"] = self.llm.invoke(msg).content
        state.setdefault("checkpoints", []).append({"step": "generate", "state": dict(state)})
        return state

    # ---------- 3) Graph bauen ----------
    def build_graph(self):
        if self.app is None:
            self.graph = StateGraph(RAGState)
            self.graph.add_node("condense", self.condense_node)
            self.graph.add_node("retrieve", self.retrieve_node)
            self.graph.add_node("generate", self.generate_node)

            self.graph.add_edge(START, "condense")
            self.graph.add_edge("condense", "retrieve")
            self.graph.add_edge("retrieve", "generate")
            self.graph.add_edge("generate", END)

            self.app = self.graph.compile()
        return self.app

# ---------- 4) Instanz erstellen ----------
# bot = RAGBot()
# app = bot.build_graph()

# ---------- 5) Ausführen ----------
# cfg = {"configurable": {"thread_id": "user-42"}}
# inp = {
#     "messages": [
#         {"role": "user", "content": "Was ist ein Node?"}
#     ]
# }
# out = app.invoke(inp,cfg)
# print(out["answer"])


# out = app.invoke({
#     "messages": [
#         {"role": "user", "content": "Was war meine Frage?"}
#     ]
# },cfg)
# print(out["answer"])


# print(out["checkpoints"])
