## **Module 3: How to secure our RAG Applications**

**Video 1: Threat Modelling our RAG Application**

* How the NovaBank case study sets up the RAG we build in this module, and why a RAG that works is not yet safe for a bank.  
* How the threat modelling framework from Module 2 (DFD, entry and exit points, trust boundaries and the capability table) finds every weak point in our RAG, and which security layer each video adds to close it.

**Video 2: Securing Vector Database \- PII Leakage**

* How raw bank documents leak customers' personal data through the vector database.  
* How a PII detection and masking layer runs on every document before ingestion, so no one's personal details end up stored in the vector database.

**Video 3: Securing Retriever \- Authentication and Authorization**

* How authentication and authorization work at production level.  
* How a pre-retrieval authorization layer uses metadata filtering to block restricted documents before the search even runs.

**Video 4: Securing User Query and LLM \- Jailbreak and Prompt Injection**

* How a jailbreak attacks the LLM directly through the user query, and how a guardrail layer between the user query and the retriever blocks it.  
* How a prompt injection attacks the LLM indirectly through a retrieved document, and how delimiters and a system prompt that marks documents as data defend against it.

**Video 5: Securing Output \- Output Guardrails**

* How an unchecked answer can carry private data, wrong facts or a tricked response straight to the bank employee.  
* How output guardrails built with NVIDIA NeMo Guardrails check every answer before the employee sees it.

**Video 6: Red Teaming our RAG Application**

* What red teaming is, and how security layers are tested at production level.  
* How to run every attack against the full secured RAG with Promptfoo, to see which security layers hold.

---

## **Module 4: How to secure our AI Agents**

**Video 1: Threat Modelling our AI Agent**

* How the doctor-booking case study sets up the agent we build in this module, and why an agent that works is not yet safe to put in front of anyone on the internet.  
* How the threat modelling framework from Module 2 (DFD, entry and exit points, trust boundaries and the capability table) finds every weak point in our agent, and which security layer each video adds to close it.

**Video 2: Securing User Query \- Jailbreak**

* How a jailbreak through the patient's message, such as a false claim to be the CEO or a doctor, makes the agent reveal other patients' bookings and cancel real ones.  
* How a jailbreak gate uses a separate guard model to check every message, so a blocked request never reaches the agent or its tools.

**Video 3: Securing Tools and MCP \- Prompt Injection**

* How  hidden instructions in a doctor's page or schedule note reach the LLM through tool and MCP results, then bias its recommendations or make it cancel bookings no patient asked to cancel.  
* How a guard model checks every tool and MCP result before the LLM reads it, and how delimiters and a system prompt that marks tool results as data add a second defense.
