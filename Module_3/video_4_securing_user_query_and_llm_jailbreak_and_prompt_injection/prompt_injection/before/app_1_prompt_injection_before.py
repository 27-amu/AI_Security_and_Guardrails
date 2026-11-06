"""
app_1_prompt_injection_before.py — BEFORE. The same flat NovaBank RAG as video_3_securing_retriever_authentication_and_authorization/before/app_1_flat_retriever.py.
The retrieved chunks are pasted into the prompt as plain text, so the model cannot tell bank data from instructions.
TKT-1058 contains a customer's email that was copied into the ticket automatically, with a fake "note for the assistant".

Build the index ONCE:   python video_4_securing_user_query_and_llm_jailbreak_and_prompt_injection/prompt_injection/before/app_1_prompt_injection_before.py --build
Run the demo:           streamlit run video_4_securing_user_query_and_llm_jailbreak_and_prompt_injection/prompt_injection/before/app_1_prompt_injection_before.py
Attack query:           What is the next step for ticket TKT-1058?
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
ROOT = HERE.parents[2]                       # Module_3/
load_dotenv(ROOT / ".env")

KB = ROOT / "bank_data" / "knowledge_base"
STORE = str(HERE.parent / "store")           # shared by before and after: same documents, only the prompt changes
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
    question: str
    chunks: list[Document]
    answer: str


def retrieve(state: QAState):
    return {"chunks": vector_db().similarity_search(state["question"], k=3)}


def generate(state: QAState):
    context = "\n\n".join(c.page_content for c in state["chunks"])          # <-- data and instructions mixed in one blob
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
def ui():
    st.title("NovaBank assistant — prompt injection BEFORE")
    question = st.chat_input("Ask a question, for example: What is the next step for ticket TKT-1058?")
    if question:
        st.chat_message("user").write(question)
        result = qa_graph.invoke({"question": question})
        st.chat_message("assistant").write(result["answer"])
        with st.expander("Chunks read from the vector database"):
            for c in result["chunks"]:
                st.caption(c.metadata["source"])
                st.code(c.page_content)


if "--build" in sys.argv:       # one-time build (the index is shared with after/)
    ingest_graph.invoke({"docs": []})
    print("Index built in", STORE)
elif "--ask" in sys.argv:       # terminal run for repeat testing: --ask "question"
    r = qa_graph.invoke({"question": sys.argv[sys.argv.index("--ask") + 1]})
    print("SOURCES:", [c.metadata["source"] for c in r["chunks"]])
    print(r["answer"])
else:
    ui()
