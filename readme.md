# Medical Drug QA Chatbot (Retrieval-Based)

This project implements a Retrieval based Question Answering system that answers user queries about medical drugs. The system combines semantic retrieval (using FAISS and MiniLM embeddings) with reranking (using BioBERT) and answer generation (using Groq-hosted LLaMA-4 models).

---

## Features

- Named Entity Recognition using SciSpaCy
- Rule-based intent and sub-intent classification
- Dense retrieval using FAISS and MiniLM-L6-v2
- Semantic reranking with Sentence-BioBERT
- Generative answer synthesis using LLaMA-4 (via Groq API)
- Evaluation metrics: Precision@3, Recall@3, F1 Score@3, Success Rate@3

---

##  Project Structure

```
.
├── Query_processing.py        # Query intent classification and NER
├── Retrieval.py               # Embedding, FAISS indexing, and reranking logic
├── Answer_Generation.py       # Prompt building and Groq LLM integration
├── Evaluation.py              # Evaluation of retrieval performance
├── flattened_drug_dataset_cleaned.csv
├── custom_drug_eval_set_id.csv
├── faiss_index.idx
├── doc_metadata.pkl
├── doc_vectors.npy
├── requirements.txt
└── README.md
```

---

##  Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate FAISS Index (only once)

```python
from Retrieval import Embed_and_FAISS
Embed_and_FAISS()
```

### 3. Run a Sample Query

```python
from Query_processing import preprocess_query
from Retrieval import Retrieval_averagedQP
from Answer_Generation import answer_generation

question = "What is the dosage for Azithromycin?"
(intent, sub_intent), entities = preprocess_query(question)
chunks = Retrieval_averagedQP(question, intent, entities)
answer = answer_generation(question, chunks)
print(answer)
```

---

## Evaluation Results

| Metric         | Value   |
|----------------|---------|
| Success Rate@3 | 87.50 % |
| Precision@3    | 53.30 % |
| Recall@3       | 76.25 % |
| F1 Score@3     | 60.96 % |

---

## Models Used

- **MiniLM-L6-v2**: For FAISS-based vector retrieval
- **Sentence-BioBERT**: For reranking candidate chunks
- **LLaMA-4** (Groq API): For final answer generation

---

## Authors

Niranjan Sathish
Graduate Student, Masters in Artificial Intelligence
Northeastern University

Hariharan Chandrasekar
Graduate Student, Masters in Artificial Intelligence
Northeastern University