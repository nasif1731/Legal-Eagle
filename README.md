# ⚖️ Legal-Eagle: AI Compliance Auditor

**Automated Contract Review & Risk Analysis.**  
*A Retrieval-Augmented Generation (RAG) system that audits complex legal agreements against 41 strict compliance rules, providing evidence, status verdicts, and remediation clauses.*

---

## 🧐 What is this?

Reviewing contracts is tedious, expensive, and prone to human error. **Legal-Eagle** automates this by treating legal compliance as a search-and-reasoning problem.

It downloads real commercial contracts (from the **CUAD dataset**), indexes them into a vector database, and uses a quantized **Mistral-7B** LLM to act as a "Strict Legal Auditor." For every rule (e.g., *"Is there a Non-Compete clause?"*), it scans the document, extracts the exact text, and judges if it **PASSES** or **FAILS**.

---

## ⚙️ System Architecture

The system relies on a locally running RAG pipeline optimized for legal text.

1. **Data Ingestion:** Downloads the **CUAD (Contract Understanding Atticus Dataset)** and selects a long-form contract (>5,000 chars) to ensure complexity.
2. **Vector Store:** Splits the contract into 1000-character chunks and embeds them using `all-MiniLM-L6-v2` into **ChromaDB**.
3. **The Auditor (LLM):** Uses **Mistral-7B-Instruct-v0.2 (GGUF)** loaded via `CTransformers` for offline, GPU-accelerated inference.
4. **Chain:** A custom LangChain prompt instructs the model to output **Status**, **Evidence**, and **Remediation**.

---

## ⚡ Quick Start

**1. Install Dependencies**  
The system requires LangChain v0.2 ecosystem and CTransformers with CUDA support.

```bash
pip install -qU langchain==0.2.0 langchain-community==0.2.0 chromadb sentence-transformers ctransformers[cuda]
```

**2. Run the Pipeline**  
Execute the script to download the data, build the database, and load the LLM.

```python
# This initializes the database and loads Mistral
vectorstore = setup_compliance_db()
```

---

## 📋 Automated Audit

The core feature is the **Batch Compliance Check**. The system iterates through **41 standard legal rules** (e.g., Governing Law, IP Ownership, Termination for Convenience) and generates a detailed CSV report.

**Example Output:**

| Rule | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| **Governing Law** | **PASS** | *"This Agreement... under the laws of Ohio..."* | None |
| **Source Code Escrow** | **FAIL** | *None* | *"Add a clause requiring Agent to deposit source code..."* |
| **Non-Compete** | **PASS** | *"Monsanto and Agent will not participate in Competitive Business..."* | None |

*The full report is saved to `compliance_audit_report.csv`.*

---

## 🚀 Run the UI

For interactive testing, the project includes a **Gradio Interface**.

```python
# Launch the web app
demo.queue().launch(share=True)
```

**How to use:**

1. Open the provided URL (e.g., `https://...gradio.live`).
2. Type a custom rule (e.g., *"The agreement must mention 'Force Majeure'."*).
3. The agent will retrieve relevant context and return a verdict.

---

## 🧠 Prompt Engineering

The "Secret Sauce" is the strict system prompt used in the `audit_chain`:

> "You are a strict Legal Compliance Auditor... Determine if the Rule is met (PASS) or violated/missing (FAIL). Extract the exact text that proves your decision (Evidence). If FAIL, suggest exactly what clause needs to be added (Remediation)."

This forces the LLM to be structured and actionable rather than conversational.

---

## 📜 Credits

* **Dataset:** The Atticus Project (CUAD)
* **Model:** Mistral AI (TheBloke Quantization)
* **Framework:** LangChain & Gradio

---

**⚖️ Disclaimer:** *This AI tool is for educational and assistance purposes only. It is not a substitute for a qualified attorney.*
