"""
app_1_output_guardrails_before.py — BEFORE. The same flat NovaBank RAG, over every document in the
knowledge base, with a two-line system prompt. 

The model's answer goes STRAIGHT to the user: nothing inspects the output, so whatever an earlier layer let through (restricted chunks, customer PII, off-policy advice) is delivered.

Build the index ONCE:        python video_5_securing_output_output_guardrails/before/app_1_output_guardrails_before.py --build
Run the demo:                streamlit run video_5_securing_output_output_guardrails/before/app_1_output_guardrails_before.py
"""

import sys
from pathlib import Path
from typing import TypedDict

import streamlit as st
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
STORE = str(HERE / "store")                   # this app has its own index folder, built from the same documents

LOADERS = {".md": lambda p: TextLoader(p, encoding="utf-8"),            # one loader per file type of the bank
           ".docx": Docx2txtLoader,
           ".html": lambda p: BSHTMLLoader(p, open_encoding="utf-8")}
FILES = sorted(p for p in KB.rglob("*") if p.suffix in LOADERS)         # every file in the knowledge base


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# A 2 liner internal-assistant prompt
SYSTEM_PROMPT = (
    "You are the assistant for Novabank whose task is to help the bank employees with their queries."
    
    "Use the given context to answer the user query." 
)

def vector_db() -> Chroma:
    return Chroma("bank_output_guardrails", OllamaEmbeddings(model="embeddinggemma:300m-qat-q4_0"), persist_directory=STORE)


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
    question: str
    history: list[tuple[str, str]]          # earlier turns as (role, text); the model sees the whole conversation
    chunks: list[Document]
    answer: str


def retrieve(state: QAState):
    return {"chunks": vector_db().similarity_search(state["question"], k=3)}     # the raw user message goes straight in


def generate(state: QAState):
    context = "\n\n".join(c.page_content for c in state["chunks"])
    
    reply = llm.invoke([("system", SYSTEM_PROMPT),
                        *state["history"],                  # past turns, without their old context
                        ("human", f"Context:\n{context}\n\nQuestion: {state['question']}")])
    
    return {"answer": reply.content}


q = StateGraph(QAState)
q.add_node("retrieve", retrieve)
q.add_node("generate", generate)
q.add_edge(START, "retrieve")
q.add_edge("retrieve", "generate")
q.add_edge("generate", END)
qa_graph = q.compile()


# ---------------- Streamlit ----------------
def show(msg: dict):
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        if msg.get("chunks"):
            with st.expander("Chunks read from the vector database"):
                for c in msg["chunks"]:
                    st.caption(c.metadata["source"])
                    st.code(c.page_content)


def ui():
    st.title("NovaBank assistant — BEFORE (no output guardrails)")
    if "messages" not in st.session_state:
        st.session_state.messages = []          # kept for this browser session only
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
    for msg in st.session_state.messages:       # Streamlit reruns the script on every message, so redraw the chat
        show(msg)

    question = st.chat_input("Ask a question, for example: How many days do we have to resolve a customer complaint?")
    if question:
        history = [("human" if m["role"] == "user" else "ai", m["text"]) for m in st.session_state.messages]
        user_msg = {"role": "user", "text": question}
        show(user_msg)
        result = qa_graph.invoke({"question": question, "history": history})
        bot_msg = {"role": "assistant", "text": result["answer"], "chunks": result["chunks"]}
        show(bot_msg)
        st.session_state.messages += [user_msg, bot_msg]


if "--build" in sys.argv:       # one-time build
    ingest_graph.invoke({"docs": []})
    print("Index built in", STORE)
else:
    ui()
