
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class ProductVectorIndex:
    def __init__(self, emb, path="../faiss_index"):
        self.emb = emb
        self.path = path
        self.vs = None

    def load_or_create(self, md_path_str="../product_faq.md"):
        # Nutze build_md_faiss_index, falls Index nicht existiert
        if not os.path.exists(self.path):
            build_md_faiss_index(md_path_str, self.path)
        else:
            print(f"Index exists: {self.path}")
        self.vs = FAISS.load_local(self.path, self.emb, allow_dangerous_deserialization=True)
        return self.vs

    def get_retriever(self, k=3):
        if not self.vs:
            raise ValueError("Index not loaded. Call load_or_create first.")
        return self.vs.as_retriever(search_kwargs={"k": k})






def build_md_faiss_index(md_path_str="../product_faq.md", index_path="../faiss_index", meta_path="../md_faiss_meta.json"):
    from pathlib import Path
    from markdown_it import MarkdownIt
    import json
    import re
    md_path = Path(md_path_str)
    raw_md = md_path.read_text(encoding="utf-8")

    def split_by_headings(md_text: str):
        parts = re.split(r'(?=^#{1,6}\s)', md_text, flags=re.MULTILINE)
        return [p.strip() for p in parts if p.strip()]

    sections = split_by_headings(raw_md)
    md = MarkdownIt()
    def md_to_text(md_chunk: str) -> str:
        html = md.render(md_chunk)
        text = re.sub(r'<[^>]+>', '', html)
        return re.sub(r'\s+\n', '\n', text).strip()

    plain_sections = [md_to_text(s) for s in sections]

    def chunk_text(text: str, max_chars=1200, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunk = text[start:end]
            chunks.append(chunk.strip())
            if end - overlap > start:
                start = end - overlap
            else:
                start = end
        return [c for c in chunks if c]

    docs = []
    for sec in plain_sections:
        paras = [p.strip() for p in re.split(r'\n{2,}', sec) if p.strip()]
        merged = "\n\n".join(paras)
        for ch in chunk_text(merged, max_chars=1000, overlap=150):
            docs.append(ch)

    metas = [{"source": str(md_path), "chunk_id": i} for i in range(len(docs))]

    from langchain_openai import OpenAIEmbeddings
    emb_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embs = emb_model.embed_documents(docs)

    faiss_vs = FAISS.from_texts(docs, emb_model)
    faiss_vs.save_local(index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "metas": metas}, f, ensure_ascii=False, indent=2)

    print(f"Index gebaut: {len(docs)} Chunks")