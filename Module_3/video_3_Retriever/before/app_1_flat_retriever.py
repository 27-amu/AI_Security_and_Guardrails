"""
app_1_flat_retriever.py — BEFORE. One vector database, one collection, no identity check in the retriever.
The user dropdown exists, but the retriever never reads it. So every user can retrieve every document.

Build the index ONCE before the demo:   python video_3_Retriever/before/app_1_flat_retriever.py --build
Run the demo:                           streamlit run video_3_Retriever/before/app_1_flat_retriever.py
"""

import sys
from pathlib import Path
from typing import TypedDict

import streamlit as st
import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import BSHTMLLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # Module_3/
load_dotenv(ROOT / ".env")

KB = ROOT / "bank_data" / "knowledge_base"
STORE = str(HERE / "store_flat")
USERS = yaml.safe_load((HERE.parent / "users.yaml").read_text(encoding="utf-8"))["users"]
LOADERS = {".md": lambda p: TextLoader(p, encoding="utf-8"),            # one loader per file type of the bank
           ".docx": Docx2txtLoader,
           ".html": lambda p: BSHTMLLoader(p, open_encoding="utf-8")}
FILES = sorted(p for p in KB.rglob("*") if p.suffix in LOADERS)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def vector_db() -> Chroma:                   
    return Chroma("bank_all", OllamaEmbeddings(model="embeddinggemma:300m-qat-q4_0"), persist_directory=STORE)


# ---------------- graph 1: ingestion ----------------
class IngestState(TypedDict):
    docs: list[Document]


def load(state: IngestState):
    docs = []
    for f in FILES:
        for d in LOADERS[f.suffix](str(f)).load():
            d.metadata = {"source": f.name}
            docs.append(d)
    return {"docs": docs}


def split(state: IngestState):
    return {"docs": RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(state["docs"])}


def index(state: IngestState):
    vector_db().delete_collection()
    vector_db().add_documents(state["docs"])         
    return {}


g = StateGraph(IngestState)
g.add_node("load", load)
g.add_node("split", split)
g.add_node("index", index)
g.add_edge(START, "load")
g.add_edge("load", "split")
g.add_edge("split", "index")
g.add_edge("index", END)
ingest_graph = g.compile()


# ---------------- graph 2: question answering ----------------
class QAState(TypedDict):
    user_id: str
    question: str
    chunks: list[Document]
    answer: str


def retrieve(state: QAState):
    return {"chunks": vector_db().similarity_search(state["question"], k=3)}     # <-- state["user_id"] is never used


def generate(state: QAState):
    context = "\n\n".join(c.page_content for c in state["chunks"])
    reply = llm.invoke(f"Use the given context to answer the user question.\n\nContext:\n{context}\n\nQuestion: {state['question']}")
    
    return {"answer": reply.content}


q = StateGraph(QAState)
q.add_node("retrieve", retrieve)
q.add_node("generate", generate)
q.add_edge(START, "retrieve")
q.add_edge("retrieve", "generate")
q.add_edge("generate", END)
qa_graph = q.compile()


# ---------------- Streamlit ----------------
def category(user_id: str) -> str:                                      # role + level, same text as in the AFTER app
    user = USERS[user_id]
    return " · ".join([user["role"]] + ([user["level"]] if "level" in user else []))


def ui():
    st.title("NovaBank assistant — BEFORE")
    user_id = st.sidebar.selectbox("Logged in as", list(USERS), index=list(USERS).index("hr_manager"), format_func=category)
    question = st.chat_input("Ask a question, for example: What is the salary band for a branch manager?")
    if question:
        st.chat_message("user").write(question)
        result = qa_graph.invoke({"user_id": user_id, "question": question})
        st.chat_message("assistant").write(result["answer"])
        with st.expander("Chunks read from the vector database"):
            for c in result["chunks"]:
                st.caption(c.metadata["source"])
                st.code(c.page_content)


if "--build" in sys.argv:       # one-time build
    ingest_graph.invoke({"docs": []})
    print("Index built in", STORE)
else:
    ui()
