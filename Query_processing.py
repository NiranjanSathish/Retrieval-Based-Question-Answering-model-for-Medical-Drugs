"""
Query Processing Pipeline for Retrieval-based QA Chatbot
========================================================

This module handles:
1. Query preprocessing
2. Intent and sub-intent classification
3. Named Entity Recognition (NER) using SciSpaCy

"""

import spacy
import re
from typing import List, Tuple

# Load pre-trained SciSpaCy model for biomedical NER
ner_model = spacy.load("en_core_sci_md")

# -------------------------------
# Rule-Based Intent Classification
# -------------------------------

def classify_intent(question: str) -> str:
    """
    Classify the user's query into a high-level intent based on keywords.
    Replace this rule-based system with ML-based intent detection for scalability.

    Parameters:
        question (str): The user's question.

    Returns:
        str: One of ['description', 'before_using', 'proper_use', 'precautions', 'side_effects']
    """
    q = question.lower()

    if re.search(r"\bwhat is\b|\bused for\b|\bdefine\b", q):
        return "description"
    elif re.search(r"\bbefore using\b|\bshould I tell\b|\bdoctor know\b", q):
        return "before_using"
    elif re.search(r"\bhow to\b|\bdosage\b|\btake\b|\binstructions\b", q):
        return "proper_use"
    elif re.search(r"\bprecaution\b|\bpregnan\b|\bbreastfeed\b|\brisk\b", q):
        return "precautions"
    elif re.search(r"\bside effect\b|\badverse\b|\bnausea\b|\bdizziness\b", q):
        return "side_effects"
    else:
        return "description"  # default fallback


# -------------------------------
# Subsection Classification
# -------------------------------

def classify_subsection(question: str) -> str:
    """
    Identify more granular subtopics within each main intent.

    Parameters:
        question (str): The user's question.

    Returns:
        str: Sub-intent such as 'more common', 'incidence not known', etc.
    """
    q = question.lower()

    if re.search(r"\bcommon side effects\b|\busual symptoms\b", q):
        return "more common"
    elif re.search(r"\bunknown\b|\brare\b|\bincidence\b", q):
        return "incidence not known"
    elif re.search(r"\bchildren\b|\bpediatric\b|\bkids\b", q):
        return "pediatric"
    elif re.search(r"\bbreastfeed\b|\bnursing\b|\blactation\b", q):
        return "breastfeeding"
    elif re.search(r"\belderly\b|\bgeriatric\b", q):
        return "geriatric"
    elif re.search(r"\binteract\b|\bcombination\b|\bcontraindications\b", q):
        return "drug interactions"
    else:
        return ""


# -------------------------------
# Named Entity Extraction
# -------------------------------

def extract_entities_spacy(question: str) -> List[str]:
    """
    Use SciSpaCy NER model to extract biomedical entities.

    Parameters:
        question (str): User query.

    Returns:
        List[str]: Unique list of extracted entities.
    """
    doc = ner_model(question)
    return list(set(ent.text for ent in doc.ents))


# -------------------------------
# Query Preprocessing Wrapper
# -------------------------------

def preprocess_query(raw_query: str) -> Tuple[Tuple[str, str], List[str]]:
    """
    Main preprocessing function that extracts:
    - Intent
    - Subsection
    - Named Entities

    Parameters:
        raw_query (str): The raw user question.

    Returns:
        Tuple[Tuple[str, str], List[str]]: ((intent, sub_intent), list of entities)
    """
    try:
        intent = classify_intent(raw_query)
        sub_intent = classify_subsection(raw_query)
        entities = extract_entities_spacy(raw_query)

        if not entities:
            print("[NER fallback] No entities found. Using raw query.")
            return (intent or "", sub_intent or ""), []

        print(f"[Query Processed] Intent = {intent} | Subsection = {sub_intent} | Entities = {entities}")
        return (intent or "", sub_intent or ""), entities

    except Exception as e:
        print(f"[Preprocessing failed] {e}")
        return ("", ""), []