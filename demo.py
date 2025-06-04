"""
Main Execution Script for Retrieval-based Medical QA Chatbot
============================================================

This script handles:
1. Query preprocessing
2. Information retrieval
3. Answer generation
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from Query_processing import preprocess_query
from Retrieval import Retrieval_averagedQP
from Answer_Generation import answer_generation
from Retrieval import Embed_and_FAISS

# -------------------------------
# Optional: Embed and Store FAISS Index
# -------------------------------
# Uncomment the below line to generate embeddings and build the FAISS index if not already done.
# Embed_and_FAISS()

# -------------------------------
# Define User Question
# -------------------------------

Question = "how much dosage of azithromycin should be taken for treatment of pharyngitis or tonsillitis?"

# -------------------------------
# Step 1: Query Preprocessing
# -------------------------------

(intent, sub_intent), entities = preprocess_query(Question)

# -------------------------------
# Step 2: Retrieve Relevant Chunks
# -------------------------------

top_chunks = Retrieval_averagedQP(Question, intent, entities, top_k=10, alpha=0.8)

# -------------------------------
# Step 3: Answer Generation
# -------------------------------

Generated_answer = answer_generation(Question, top_chunks, top_k=3)

# -------------------------------
# Display Generated Answer
# -------------------------------

print("Generated Answer:", Generated_answer)
