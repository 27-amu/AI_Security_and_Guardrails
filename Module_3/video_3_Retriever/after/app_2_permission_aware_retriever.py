"""
app_2_permission_aware_retriever.py — AFTER. The same app as app_1_flat_retriever.py, with ONE security layer BEFORE retrieval:
  1. CATALOG       — catalog.yaml says, for each folder (and each support ticket), who may read it:
                     `allowed_roles` and `level`. It stands in for the bank's document system and CRM.
  2. METADATA      — at ingestion every chunk stores `folder`, `allowed_roles` and `level`.
  3. PRE-retrieval — the retriever builds a filter from the user (users.yaml: role, level) and sends it with the search.
                     Only chunks that match are candidates, so the LLM never sees the others.
The prompt only words the reply when the allowed chunks hold no answer. It is NOT the security control.
The message is neutral on purpose: it must not reveal that restricted data exists.

Build the index ONCE before the demo:   python video_3_Retriever/after/app_2_permission_aware_retriever.py --build
Run the demo:                           streamlit run video_3_Retriever/after/app_2_permission_aware_retriever.py
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
STORE = str(HERE / "store_catalog")

USERS = yaml.safe_load((HERE.parent / "users.yaml").read_text(encoding="utf-8"))["users"]
CATALOG = yaml.safe_load((HERE.parent / "catalog.yaml").read_text(encoding="utf-8"))

LOADERS = {".md": lambda p: TextLoader(p, encoding="utf-8"),            # one loader per file type of the bank
           ".docx": Docx2txtLoader,
           ".html": lambda p: BSHTMLLoader(p, open_encoding="utf-8")}
FILES = sorted(p for p in KB.rglob("*") if p.suffix in LOADERS)


NOT_FOUND = "I could not find this in the documents available to you."
K = 3                                                                   # chunks sent to the LLM

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def vector_db() -> Chroma:                                              # ONE collection holds every chunk
    return Chroma("bank_all", OllamaEmbeddings(model="embeddinggemma:300m-qat-q4_0"), persist_directory=STORE)


def labels_of(path: Path) -> dict:                                      # catalog.yaml -> metadata of one file
    folder = path.relative_to(KB).parent.as_posix()                     # retail/hr_confidential
    rule = CATALOG["folders"][folder]
    level = rule.get("level") or CATALOG["tickets"][path.name]          # tickets: the level comes from the CRM field
    return {"source": path.name, "folder": folder, "allowed_roles": list(rule["allowed_roles"]), "level": level}



# ---------------- graph 1: ingestion ----------------
class IngestState(TypedDict):
    docs: list[Document]


def load(state: IngestState):
    docs = []
    for p in FILES:
        for d in LOADERS[p.suffix](str(p)).load():
            d.metadata = labels_of(p)                                   # NEW: the access labels travel with the data
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
    where: dict
    chunks: list[Document]
    answer: str


def access_filter(user_id: str) -> dict:
    """Built ONLY from the verified user, never from the question or the LLM.
    Allow-list operators only ($contains, $in). Never $ne, $nin or $not_contains: they also match chunks
    that miss the key, so untagged chunks would leak."""
    user = USERS[user_id]
    levels = ["all"] + ([user["level"]] if "level" in user else [])
    return {"$and": [{"allowed_roles": {"$contains": user["role"]}},
                     {"level": {"$in": levels}}]}


def retrieve(state: QAState):
    # PRE-retrieval: the filter travels with the search. Only matching chunks can be candidates.
    where = access_filter(state["user_id"])
    return {"where": where, "chunks": vector_db().similarity_search(state["question"], k=K, filter=where)}


def generate(state: QAState):
    context = "\n\n".join(c.page_content for c in state["chunks"])
    reply = llm.invoke(f"Use the given context to answer the user question. If the context does not contain the answer, "
                       f"reply exactly: {NOT_FOUND}\n\nContext:\n{context}\n\nQuestion: {state['question']}")
    return {"answer": reply.content}


q = StateGraph(QAState)
q.add_node("retrieve", retrieve)
q.add_node("generate", generate)
q.add_edge(START, "retrieve")
q.add_edge("retrieve", "generate")
q.add_edge("generate", END)
qa_graph = q.compile()


# ---------------- Streamlit ----------------
def category(user_id: str) -> str:                                      # role + level, for example customer_support_agent · new_joiner
    user = USERS[user_id]
    return " · ".join([user["role"]] + ([user["level"]] if "level" in user else []))


def ui():
    st.title("NovaBank assistant — AFTER (permission-aware retriever)")
    user_id = st.sidebar.selectbox("Logged in as", list(USERS), index=list(USERS).index("hr_manager"), format_func=category)
    user = USERS[user_id]
    if "level" in user:                                                 # hr_manager and ceo have no level: no caption
        st.sidebar.caption(f"role: {user['role']} · level: {user['level']}")
    question = st.chat_input("Ask a question, for example: What is the salary band for a branch manager?")
    if question:
        st.chat_message("user").write(question)
        result = qa_graph.invoke({"user_id": user_id, "question": question})
        st.chat_message("assistant").write(result["answer"])
        with st.expander("Retrieval audit: what the LLM received"):
            st.write("Filter sent with the search:")
            st.json(result["where"])
            for c in result["chunks"]:
                m = c.metadata
                st.caption(f"{m['source']} · {m['folder']} · {m['allowed_roles']} · {m['level']}")
                st.code(c.page_content)


if "--build" in sys.argv:       # one-time build, run before the demo
    ingest_graph.invoke({"docs": []})
    print("Index built in", STORE)
elif __name__ == "__main__":    # streamlit runs the file as __main__; an import does not start the page
    ui()
