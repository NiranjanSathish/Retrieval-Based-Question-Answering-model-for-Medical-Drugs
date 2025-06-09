"""
Answer Generation Module for Retrieval-based Medical QA Chatbot
=================================================================

This module handles:
1. Building prompts for LLMs
2. Querying the Groq API with selected context
3. Generating a final answer based on retrieved chunks
"""

from openai import OpenAI
from Retrieval import Retrieval_averagedQP
import streamlit as st

# -------------------------------
# Groq API Client Setup
# -------------------------------

try:
    client = OpenAI(
        api_key=st.secrets["groq_api_key"],
        base_url="https://api.groq.com/openai/v1"
    )

    print("✅ Groq client initialized")
except Exception as e:
    print(f"❌ Failed to initialize Groq client: {e}")
    raise


# -------------------------------
# Function: Query Groq API
# -------------------------------

def query_groq(prompt, model="meta-llama/llama-4-scout-17b-16e-instruct", max_tokens=300):
    """
    Sends a prompt to Groq API and returns the generated response.

    Parameters:
        prompt (str): The text prompt for the model.
        model (str): Model name deployed on Groq API.
        max_tokens (int): Maximum tokens allowed in the output.

    Returns:
        str: Model-generated response text.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a biomedical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

# -------------------------------
# Function: Build Prompt
# -------------------------------

def build_prompt(question, context):
    """
    Constructs a prompt for the model combining the user question and retrieved context.

    Parameters:
        question (str): User's question.
        context (str): Retrieved relevant text chunks.

    Returns:
        str: Complete prompt text.
    """
    return f"""Strictly based on the following information, answer the question: {question}
Do not explain the context, just provide a direct answer.

Context:
{context}
"""

# -------------------------------
# Function: Answer Generation
# -------------------------------

def answer_generation(question, top_chunks, top_k=3):
    """
    Generates an answer based on retrieved top chunks.

    Parameters:
        question (str): User's question.
        top_chunks (DataFrame): Retrieved top chunks with context.
        top_k (int): Number of top chunks to use for answer generation.

    Returns:
        str: Final generated answer.
    """
    # Select top-k chunks
    top_chunks = top_chunks.head(top_k)
    print("[Answer Generation] Top chunks selected for generation.")

    # Join context
    context = "\n".join(top_chunks["chunk_text"].tolist())

    # Build prompt and query Groq
    prompt = build_prompt(question, context)
    answer = query_groq(prompt)

    return answer

# -------------------------------
# Example Usage (Uncomment to Test)
# -------------------------------

# question = "How is Aztreonam inhalation used?"
# answer = answer_generation(question, top_chunks)
# print("Generated Answer:", answer)
