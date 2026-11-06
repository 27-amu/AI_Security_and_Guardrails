"""
app_1_raw_rag.py — BEFORE. A small RAG app (LangGraph + Chroma + Streamlit) with NO PII check.
The documents go straight into the vector database, so the database holds names, phones, PAN and card numbers.

Build the index ONCE before the demo:   python video_2_PII_on_vector_database/before/app_1_raw_rag.py --build
Run the demo:                           streamlit run video_2_PII_on_vector_database/before/app_1_raw_rag.py
"""

import sys
from pathlib import Path
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       
load_dotenv(ROOT / ".env")

KB = ROOT / "bank_data" / "knowledge_base"
FILES = [*sorted((KB / "retail/support_tickets").glob("*.md")), KB / "wealth/clients/hni_client_review_q3.md",
         KB / "retail/policies/complaint_handling_sop.md", KB / "retail/policies/kyc_account_opening.md"]
STORE = str(HERE / "store") 

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def vector_db() -> Chroma:
    return Chroma("bank_raw", OllamaEmbeddings(model="embeddinggemma:300m-qat-q4_0"), persist_directory=STORE)


# ---------------- graph 1: ingestion ----------------
class IngestState(TypedDict):
    docs: list[Document]


def load(state: IngestState):
    return {"docs": [Document(page_content=f.read_text(encoding="utf-8"), metadata={"source": f.name}) for f in FILES]}


def split(state: IngestState):
    return {"docs": RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(state["docs"])}


def index(state: IngestState):
    vector_db().delete_collection()                  # start clean
    vector_db().add_documents(state["docs"])         # <-- raw text, PII included
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
    chunks: list[Document]
    answer: str


def retrieve(state: QAState):
    return {"chunks": vector_db().similarity_search(state["question"], k=4)}


def generate(state: QAState):
    context = "\n\n".join(c.page_content for c in state["chunks"])
    reply = llm.invoke(f"Answer the question using only the context.\n\nContext:\n{context}\n\nQuestion: {state['question']}")
    return {"answer": reply.content}


q = StateGraph(QAState)
q.add_node("retrieve", retrieve)
q.add_node("generate", generate)
q.add_edge(START, "retrieve")
q.add_edge("retrieve", "generate")
q.add_edge("generate", END)
qa_graph = q.compile()


# ---------------- Streamlit ----------------
def ui():
    st.title("NovaBank assistant — BEFORE")
    question = st.chat_input("Ask a question, for example: What is Priya Malhotra's phone number and PAN?")
    if question:
        st.chat_message("user").write(question)
        result = qa_graph.invoke({"question": question})
        st.chat_message("assistant").write(result["answer"])
        with st.expander("Chunks read from the vector database"):
            for c in result["chunks"]:
                st.caption(c.metadata["source"])
                st.code(c.page_content)


if "--build" in sys.argv:       # one-time build, run before the demo
    ingest_graph.invoke({"docs": []})
    print("Index built in", STORE)
else:
    ui()
