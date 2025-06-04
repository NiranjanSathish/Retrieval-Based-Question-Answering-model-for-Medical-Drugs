"""
Retrieval and FAISS Embedding Module for Medical QA Chatbot
============================================================

This module handles:
1. Embedding documents
2. Building and saving FAISS index
3. Retrieval with initial FAISS search + reranking using BioBERT similarity
"""

import faiss
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import normalize
from Query_processing import preprocess_query
import os

# -------------------------------
# File Paths
# -------------------------------

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Absolute paths for dataset and index files
csv_path = os.path.join(script_dir, 'flattened_drug_dataset_cleaned.csv')
faiss_index_path = os.path.join(script_dir, 'faiss_index.idx')
doc_metadata_path = os.path.join(script_dir, 'doc_metadata.pkl')
doc_vectors_path = os.path.join(script_dir, 'doc_vectors.npy')

# Load the dataset
df = pd.read_csv(csv_path).dropna(subset=['chunk_text'])

# -------------------------------
# Model Initialization
# -------------------------------

fast_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
biobert = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

# -------------------------------
# Function: Embed and Build FAISS Index
# -------------------------------

def Embed_and_FAISS():
    """
    Embeds the drug dataset and builds a FAISS index for fast retrieval.
    Saves the index, metadata, and document vectors to disk.
    """
    print("Embedding document chunks using fast embedder...")

    # Build full context strings
    df['full_text'] = df.apply(lambda x: f"{x['drug_name']} | {x['section']} > {x['subsection']} | {x['chunk_text']}", axis=1)

    full_texts = df['full_text'].tolist()
    doc_embeddings = fast_embedder.encode(full_texts, convert_to_numpy=True, show_progress_bar=True)

    # Normalize embeddings and build index
    doc_embeddings = normalize(doc_embeddings, axis=1, norm='l2')
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(doc_embeddings)

    # Save index and metadata
    faiss.write_index(index, faiss_index_path)
    df.to_pickle(doc_metadata_path)
    np.save(doc_vectors_path, doc_embeddings)

    print("FAISS index built and saved successfully.")

# -------------------------------
# Function: Retrieve with Context and Averaged Embeddings
# -------------------------------

def retrieve_with_context_averagedembeddings(query, top_k=10, predicted_intent=None, detected_entities=None, alpha=0.8):
    """
    Retrieve top chunks using FAISS followed by reranking with BioBERT similarity.

    Parameters:
        query (str): User query text.
        top_k (int): Number of top results to retrieve.
        predicted_intent (str, optional): Detected intent to adjust retrieval.
        detected_entities (list, optional): List of named entities.
        alpha (float): Weight for combining query and intent embeddings.

    Returns:
        pd.DataFrame: Retrieved chunks with metadata and reranked scores.
    """
    print(f"[Retrieval Pipeline Started] Query: {query}")

    # Embed and normalize the query
    query_vec = fast_embedder.encode([query], convert_to_numpy=True)

    if predicted_intent:
        intent_vec = fast_embedder.encode([predicted_intent], convert_to_numpy=True)
        query_vec = normalize((alpha * query_vec + (1 - alpha) * intent_vec), axis=1)

    # Load FAISS index and search
    index = faiss.read_index(faiss_index_path)
    D, I = index.search(query_vec, top_k)

    df_meta = pd.read_pickle(doc_metadata_path)
    retrieved_df = df_meta.loc[I[0]].copy()
    retrieved_df['faiss_score'] = D[0]

    # BioBERT reranking
    query_emb = biobert.encode(query, convert_to_tensor=True)
    chunk_embs = biobert.encode(retrieved_df['full_text'].tolist(), convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_emb, chunk_embs)[0]
    reranked_idx = torch.argsort(cos_scores, descending=True)

    # Boost scores based on intent, subsection match, or entity presence
    results = []
    for idx in reranked_idx:
        idx = int(idx)
        row = retrieved_df.iloc[idx]
        score = cos_scores[idx].item()

        section = row['section'][0] if isinstance(row['section'], tuple) else row['section']
        subsection = row['subsection'][0] if isinstance(row['subsection'], tuple) else row['subsection']
        if isinstance(predicted_intent, tuple):
            predicted_intent = predicted_intent[0]

        if predicted_intent and section.strip().lower() == predicted_intent.strip().lower():
            score += 0.05
        if predicted_intent and predicted_intent.lower() in subsection.strip().lower():
            score += 0.03
        if detected_entities:
            if any(ent.lower() in row['chunk_text'].lower() for ent in detected_entities):
                score += 0.1

        results.append({
            'chunk_id': row['chunk_id'],
            'drug_name': row['drug_name'],
            'section': row['section'],
            'subsection': row['subsection'],
            'chunk_text': row['chunk_text'],
            'faiss_score': row['faiss_score'],
            'semantic_similarity_score': score
        })

    return pd.DataFrame(results)

# -------------------------------
# Function: Retrieval Wrapper
# -------------------------------

def Retrieval_averagedQP(raw_query, intent, entities, top_k=10, alpha=0.8):
    """
    Wrapper to retrieve top-k chunks given a raw user query.

    Parameters:
        raw_query (str): The user query.
        intent (str): Predicted intent from query processing.
        entities (list): Detected biomedical entities.
        top_k (int): Number of top results to return.
        alpha (float): Weighting between query and intent embeddings.

    Returns:
        pd.DataFrame: Top retrieved chunks with scores.
    """
    results_df = retrieve_with_context_averagedembeddings(
        raw_query,
        top_k=top_k,
        predicted_intent=intent,
        detected_entities=entities,
        alpha=alpha
    )
    return results_df[['chunk_id', 'drug_name', 'section', 'subsection', 'chunk_text', 'faiss_score', 'semantic_similarity_score']]
