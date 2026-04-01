# -*- coding: utf-8 -*-


import os
import json
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# Local paths for the NTU Cluster environment
DB_PATH = "./pubmedbert_vector_store/"

# Leakage-free corpora
MAIN_BIO_PATH = "./biomedical_rag_corpus_chunked.jsonl"
SUPP_BIO_PATH = "./biomedical_rag_2_chunked.jsonl"

def process_jsonl_corpus(file_path, limit=None):
    """Loads custom JSONL datasets and safely formats metadata."""
    print(f"Loading custom corpus from {file_path}...", flush=True)
    docs, metadatas, ids = [], [], []

    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}. Skipping.", flush=True)
        return docs, metadatas, ids

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            row = json.loads(line)

            # Extract text
            docs.append(row["text"])

            # Flatten the nested "meta" dictionary safely for ChromaDB
            meta_dict = {
                "source": row.get("source", "unknown"),
                "title": str(row.get("title", ""))[:100]  # Truncate title just in case
            }

            # String any extra metadata
            for k, v in row.get("meta", {}).items():
                meta_dict[k] = str(v)

            metadatas.append(meta_dict)

            # Use the pre-computed chunk_id
            prefix = row.get("source", "doc").replace("/", "_").replace(" ", "")
            chunk_id = row.get("chunk_id", f"chk_{i}")
            ids.append(f"{prefix}_{chunk_id}")

    return docs, metadatas, ids

def build_anatomical_vector_store():
    print("Initializing ChromaDB client...", flush=True)
    client = chromadb.PersistentClient(path=DB_PATH)

    print("Loading pritamdeka/S-PubMedBert-MS-MARCO embedding model on CUDA...", flush=True)
    
    # ---------------------------------------------------------
    # Explicitly map the embedding function to the GPU
    # ---------------------------------------------------------
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO",
        device="cuda" 
    )

    # Creating the collection explicitly for the 768-dimensional vectors
    collection = client.get_or_create_collection(
        name="biomedical_knowledge_base_pubmedbert",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )

    all_docs, all_metadatas, all_ids = [], [], []

    try:
        # 1. Main Biomedical Corpus
        d, m, i = process_jsonl_corpus(MAIN_BIO_PATH)
        all_docs.extend(d); all_metadatas.extend(m); all_ids.extend(i)

        # 2. Supplementary Anatomical Corpus
        d, m, i = process_jsonl_corpus(SUPP_BIO_PATH)
        all_docs.extend(d); all_metadatas.extend(m); all_ids.extend(i)

    except Exception as e:
        print(f"\nError loading datasets. Details: {e}", flush=True)
        return

    batch_size = 5000
    total_docs = len(all_docs)
    print(f"\nTotal documents ready for ingestion: {total_docs}", flush=True)

    if total_docs == 0:
        print("No documents were loaded. Ensure JSONL paths are correct.", flush=True)
        return

    print("Embedding and adding to ChromaDB...", flush=True)

    # tqdm will track the batches and calculate ETA
    for idx in tqdm(range(0, total_docs, batch_size), desc="Ingesting Batches"):
        end_idx = min(idx + batch_size, total_docs)
        collection.add(
            documents=all_docs[idx:end_idx],
            metadatas=all_metadatas[idx:end_idx],
            ids=all_ids[idx:end_idx]
        )

    print(f"\nSuccess! Compiled at {DB_PATH}", flush=True)

if __name__ == "__main__":
    build_anatomical_vector_store()