"""
Evaluation Script for Retrieval-based QA Chatbot
=================================================

This module handles:
1. Loading evaluation questions and expected chunk IDs
2. Preprocessing queries and retrieving top chunks
3. Calculating Precision@3, Recall@3, F1-Score@3, and Success Rate@3
"""

import pandas as pd
from Query_processing import preprocess_query
from Retrieval import Retrieval_averagedQP
import os

# -------------------------------
# File Paths
# -------------------------------

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to evaluation dataset
csv_path = os.path.join(script_dir, 'custom_drug_eval_set_id.csv')

# -------------------------------
# Load Evaluation Dataset
# -------------------------------

df = pd.read_csv(csv_path)

# -------------------------------
# Evaluation Storage
# -------------------------------

all_precisions = []
all_recalls = []
all_f1s = []
all_successes = []

# -------------------------------
# Evaluation Loop
# -------------------------------

for _, row in df.iterrows():
    question = row['question']
    expected_ids = set(map(int, filter(None, str(row['relevant_chunk']).split(';'))))

    print(f"\n[Evaluation] Question: {question}")
    print(f"[Expected Chunk IDs] {expected_ids}")

    # Preprocess the query
    intent, entities = preprocess_query(question)

    # Retrieve top-k chunk predictions
    retrieved_df = Retrieval_averagedQP(question, intent, entities, top_k=10, alpha=0.8)
    retrieved_df = retrieved_df.head(3)  # Limit to top 3 results
    retrieved_ids = set(retrieved_df['chunk_id'].astype(int).tolist())

    print(f"[Retrieved Chunk IDs] {retrieved_ids}")

    # Evaluation Metrics Calculation
    tp = len(retrieved_ids & expected_ids)
    fp = len(retrieved_ids - expected_ids)
    fn = len(expected_ids - retrieved_ids)

    print(f"[Metrics] TP: {tp}, FP: {fp}, FN: {fn}")

    success = 1 if tp > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)
    all_successes.append(success)

# -------------------------------
# Aggregate Results
# -------------------------------

mean_precision = sum(all_precisions) / len(all_precisions)
mean_recall = sum(all_recalls) / len(all_recalls)
mean_f1 = sum(all_f1s) / len(all_f1s)
mean_success = sum(all_successes) / len(all_successes)

# -------------------------------
# Display Final Metrics
# -------------------------------

print("\n========= Final Evaluation Metrics =========")
print(f"Success Rate@3: {mean_success:.4f}")
print(f"Precision@3:    {mean_precision:.4f}")
print(f"Recall@3:       {mean_recall:.4f}")
print(f"F1 Score@3:     {mean_f1:.4f}")
