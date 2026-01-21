import os
import shutil
import json
import requests
import zipfile
import io
import random
import time
import pandas as pd
import gradio as gr

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import CTransformers
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "cuad_data")
DB_PATH = os.path.join(BASE_DIR, "compliance_db")
CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


LLM_CONFIG = {
    'max_new_tokens': 512, 
    'temperature': 0.0, 
    'context_length': 2048, 
    'gpu_layers': 0 
}


def setup_compliance_db():
    
    json_path = os.path.join(DATA_DIR, "test.json")
    
    if not os.path.exists(json_path):
        print(f"Downloading CUAD dataset from {CUAD_URL}...")
        print("This might take a minute...")
        try:
            r = requests.get(CUAD_URL)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(DATA_DIR)
            print("Download and extraction complete.")
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    else:
        print("Dataset already exists locally. Skipping download.")

    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 2. Pick a Random Contract
    print("Picking a random contract for analysis...")
    contract_data = None
    max_retries = 50
    
    # Try to find a nice long contract
    for _ in range(max_retries):
        rand_idx = random.randint(0, len(raw_data['data']) - 1)
        candidate = raw_data['data'][rand_idx]
        text_content = candidate['paragraphs'][0]['context']
        if len(text_content) > 5000:
            contract_data = candidate
            print(f"Selected document index: {rand_idx}")
            break
    
    if not contract_data:
        contract_data = raw_data['data'][10] # Fallback

    contract_title = contract_data.get('title', 'Unknown Contract')
    contract_text = contract_data['paragraphs'][0]['context']
    
    print(f"Analyzing Contract: {contract_title}")
    
    # 3. Vector Database (Rebuild every time we pick a new contract)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH) # Clear old DB to focus on this specific contract

    doc = Document(page_content=contract_text, metadata={"source": contract_title})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents([doc])

    print("Generating embeddings (CPU)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    print("Vector store ready.")
    
    return vectorstore, contract_title

# Initialize DB on startup
vectorstore, current_contract_title = setup_compliance_db()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- 2. LOAD LLM (CPU) ---
print("Loading Mistral-7B (CPU Mode)...")
llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    config=LLM_CONFIG
)

# --- 3. CREATE CHAIN ---
system_prompt = """You are a strict Legal Compliance Auditor. 
Your task is to check if the Document Context complies with the specific Rule.

Instructions:
1. Analyze the Context.
2. Determine if the Rule is met (PASS) or violated/missing (FAIL).
3. Extract the exact text that proves your decision (Evidence).
4. If FAIL, suggest exactly what clause needs to be added (Remediation).

Output format:
Status: [PASS or FAIL]
Evidence: [Quote from text or "None"]
Remediation: [Suggestion or "None"]

Rule to Check: {input}

Document Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI legal assistant."),
    ("human", system_prompt),
])

audit_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# --- 4. GRADIO INTERFACE ---
def audit_rule(rule_text):
    if not rule_text.strip():
        return "Please enter a rule."
    
    # Generator for real-time feedback
    yield f"🕵️ Searching contract '{current_contract_title}' for rule: '{rule_text}'..."
    
    try:
        response = audit_chain.invoke({"input": rule_text})
        raw_answer = response["answer"]
        
        formatted_output = f"""
### 📋 Audit Result
**Contract:** {current_contract_title}
**Rule:** {rule_text}

---
{raw_answer}
---
"""
        yield formatted_output
    except Exception as e:
        yield f"❌ Error: {str(e)}"

# Custom Theme
theme = gr.themes.Soft(primary_hue="slate", secondary_hue="blue")

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# ⚖️ Local AI Policy Compliance Auditor")
    gr.Markdown(f"**Currently Analyzing:** `{current_contract_title}`")
    gr.Markdown("This runs entirely on your CPU using Mistral-7B and RAG.")
    
    with gr.Row():
        with gr.Column(scale=4):
            rule_input = gr.Textbox(
                label="Enter Compliance Rule", 
                placeholder="e.g., 'The agreement must have a Force Majeure clause.'",
                lines=2
            )
        with gr.Column(scale=1):
            check_btn = gr.Button("Check Compliance", variant="primary")
    
    output_area = gr.Markdown(label="Audit Report")
    
    # Pre-defined buttons
    gr.Examples(
        examples=[
            ["The agreement must specify the Governing Law."],
            ["There must be a Non-Compete clause."],
            ["Payment terms must be Net 30 days."],
            ["Assignment to third parties is prohibited without consent."]
        ],
        inputs=rule_input
    )

    check_btn.click(fn=audit_rule, inputs=rule_input, outputs=output_area)
    rule_input.submit(fn=audit_rule, inputs=rule_input, outputs=output_area)

if __name__ == "__main__":
    print("🚀 Launching Interface...")
    demo.queue().launch() # Removed 'share=True' for local speed, add back if you want a public link