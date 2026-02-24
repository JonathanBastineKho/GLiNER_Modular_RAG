# Modular RAG for GLiNER

This project extends [GLiNER](https://github.com/urchade/GLiNER) with a **Modular RAG pipeline** that injects retrieved knowledge directly into GLiNER

---

## Project Structure

```text
/
│
├── src/
│   ├── rag/
│   │   ├── build_db.py         # Ingest raw data into the vector store
│   │   └── retriever.py        # Retriever classcontext strings
│   │
│   ├── models/
│   │   ├── gliner_base.py      # Vanilla GLiNER wrapper (no modifications)
│   │   ├── gliner_rag.py       # Modified GLiNER
|   |   └──  
│   │
│   └── components/
│       ├── cross_attention.py  # Standalone nn.Module: cross-attention block
│       ├── gate.py             # Standalone nn.Module: gating mechanism
│
├── scripts/
│   ├── train.py                # Fine-tunes GLiNER + Cross-Attn + Gate 
│   └── evaluate.py             # Evaluates all baselines and the modified GLiNER
│
│
├── data/                       # Gitignored — generated locally
│   ├── raw/                    # Raw CSV/JSONL knowledge base files
│   └── vector_store/           # Compiled e.g ChromaDB
│
├── notebooks/
│   └── exploration.ipynb       # Sanity checks and quick prototyping
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Build the Vector Database
Place your raw knowledge base file (`.csv` or `.jsonl`) in `data/raw/`, then run:
```bash
python src/rag/build_db.py
```
This embeds all documents and saves the vector store to `data/vector_store/`.

### 3. Train the modified GLiNER
```bash
python scripts/train.py
```
This only trains the **Cross-Attention + Gate** model. Baseline models do not require training (they are evaluated zero-shot or with frozen weights).

### 4. Evaluate all baselines
```bash
python scripts/evaluate.py --model base              # Vanilla GLiNER
python scripts/evaluate.py --model concat            # GLiNER + RAG concatenation
python scripts/evaluate.py --model cross_attn        # Cross-Attention
```