"""
app_2_output_guardrails_after.py — AFTER. 

The SAME RAG as BEFORE (same documents, same prompt, same model). 

The only change: the model's answer now passes through NVIDIA NeMo Guardrails OUTPUT rails - before the user sees it.

Output rails, in order:

  1. self check facts          (answer vs retrieved chunks)
  2. self check output         (answer vs NovaBank policy)

Both are LLM-as-a-judge rails built into NeMo Guardrails.

Build its own index ONCE:   python video_5_output_component/after/app_2_output_guardrails_after.py --build
Run the demo:               streamlit run video_5_output_component/after/app_2_output_guardrails_after.py
"""

import sys
from pathlib import Path
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import BSHTMLLoader, Docx2txtLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import GenerationLogOptions, GenerationOptions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # Module_3/
load_dotenv(ROOT / ".env")

KB = ROOT / "bank_data" / "knowledge_base"
STORE = str(HERE / "store")                   # this app has its own index folder, built from the same documents

LOADERS = {".md": lambda p: TextLoader(p, encoding="utf-8"),            # one loader per file type of the bank
           ".docx": Docx2txtLoader,
           ".html": lambda p: BSHTMLLoader(p, open_encoding="utf-8")}
FILES = sorted(p for p in KB.rglob("*") if p.suffix in LOADERS)         # every file in the knowledge base

# A 2 liner internal-assistant prompt
SYSTEM_PROMPT = (
    "You are the assistant for Novabank whose task is to help the bank employees with their queries."
    "Use the given context to answer the user query."
)

@st.cache_resource                           # Streamlit reruns this script on every message; load NeMo only once
def _load_rails(config_version: tuple) -> LLMRails:     # config_version is only the cache key: a changed file means a reload
    return LLMRails(RailsConfig.from_path(str(HERE / "config")))      # config.yml + rails.co + prompts.yml


def get_rails() -> LLMRails:
    return _load_rails(tuple(p.stat().st_mtime for p in sorted((HERE / "config").iterdir())))


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


# (NeMo flow name, label for the sidebar, task name of its judge call)
GUARDRAILS = [("check answer grounded", "Self check facts", "self_check_facts"),
              ("self check output", "Self check output", "self_check_output")]
ICON = {"passed": "✅", "blocked": "⛔", "not run": "⏸️"}


def render_checks(checks):
    for c in checks:
        st.markdown(f"{ICON[c['status']]} **{c['name']}**: {c['status']}")
        if c["judge"]:
            st.caption("Judge: " + c["judge"].replace("\n", "  \n"))          # verdict, then the judge's reason


def show_checks(box, checks):
    with box.container():
        if not checks:
            st.caption("Ask a question to see the verdicts.")
        render_checks(checks)


# ---------------- graph 2: question answering ----------------
class QAState(TypedDict):
    question: str
    history: list[tuple[str, str]]          # earlier turns as (role, text); the model sees the whole conversation
    chunks: list[Document]
    answer: str
    fired: list[str]                        # the output rail that blocked the answer
    raw: str                                # what the model said before the rails (demo view only)
    checks: list[dict]                      # one result per output guardrail, shown in the sidebar


def retrieve(state: QAState):
    return {"chunks": vector_db().similarity_search(state["question"], k=3)}     # the raw user message goes straight in


def generate(state: QAState):
    context = "\n\n".join(c.page_content for c in state["chunks"])
    
    messages = [{"role": "context", "content": {"relevant_chunks": context}},      # evidence for the self check facts rail
                {"role": "system", "content": SYSTEM_PROMPT},
                *[{"role": "user" if r == "human" else "assistant", "content": t} for r, t in state["history"]],
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['question']}"}]

    # The LLM call now goes through NeMo Guardrails: model -> output rails -> (possibly changed) answer.
    res = get_rails().generate(messages=messages, options=GenerationOptions(
        log=GenerationLogOptions(activated_rails=True, llm_calls=True)))
    
    answer = res.response[0]["content"]
    
    raw = next((c.completion for c in res.log.llm_calls or [] if c.task == "general"), "")
    
    fired = [r.name for r in res.log.activated_rails if r.type == "output" and r.stop]       # the rail that blocked
    
    ran = {r.name: r for r in res.log.activated_rails if r.type == "output"}
    
    judged = {c.task: (c.completion or "").strip() for c in res.log.llm_calls or []}       # raw yes/no from each judge
    
    checks = []
    
    for flow, label, task in GUARDRAILS:
        r = ran.get(flow)
        
        status = "not run" if r is None else ("blocked" if r.stop else "passed")
         
        # facts blocking skips the policy check
        checks.append({"name": label, "status": status, "judge": judged.get(task, "")})
        
    return {"answer": answer, "fired": fired, "raw": raw, "checks": checks}


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
        if msg.get("checks"):
            with st.expander("Output guardrail results", expanded=True):
                render_checks(msg["checks"])
        if msg.get("fired"):
            st.warning("Output rail(s) acted: " + ", ".join(msg["fired"]))
            with st.expander("Demo view: what the model said BEFORE the rails"):
                st.code(msg["raw"])
        if msg.get("chunks"):
            with st.expander("Chunks read from the vector database"):
                for c in msg["chunks"]:
                    st.caption(c.metadata["source"])
                    st.code(c.page_content)


def ui():
    st.title("NovaBank assistant — AFTER (NeMo output rails)")
    if "messages" not in st.session_state:
        st.session_state.messages = []          # kept for this browser session only
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []

    st.sidebar.subheader("Output guardrails")
    st.sidebar.markdown("1. **Self check facts**: the answer must be supported by the retrieved chunks\n"
                        "2. **Self check output**: the answer must follow NovaBank policy")
    st.sidebar.subheader("Result for the latest answer")
    box = st.sidebar.empty()                    # filled below, after the answer is known
    last = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
    show_checks(box, last["checks"] if last else [])

    for msg in st.session_state.messages:       # Streamlit reruns the script on every message, so redraw the chat
        show(msg)

    question = st.chat_input("Ask a question, for example: How many days do we have to resolve a customer complaint?")
    if question:
        history = [("human" if m["role"] == "user" else "ai", m["text"]) for m in st.session_state.messages]
        user_msg = {"role": "user", "text": question}
        show(user_msg)
        result = qa_graph.invoke({"question": question, "history": history})
        bot_msg = {"role": "assistant", "text": result["answer"], "chunks": result["chunks"],
                   "fired": result["fired"], "raw": result["raw"], "checks": result["checks"]}
        show(bot_msg)
        show_checks(box, bot_msg["checks"])
        st.session_state.messages += [user_msg, bot_msg]


if "--build" in sys.argv:       # one-time build
    ingest_graph.invoke({"docs": []})
    print("Index built in", STORE)
else:
    ui()
