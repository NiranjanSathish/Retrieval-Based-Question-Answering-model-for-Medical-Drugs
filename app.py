# import streamlit as st
# import pandas as pd

# from Query_processing import preprocess_query
# from Retrieval import Retrieval_averagedQP
# from Answer_Generation import answer_generation

# # -------------------------------
# # Streamlit Setup
# # -------------------------------
# st.set_page_config(page_title="Drug QA Assistant", layout="wide")
# st.title("💊 Medical Drug QA Chatbot")
# st.markdown("Ask a question about any drug. The system will retrieve relevant information and generate an answer using LLaMA-4.")

# # -------------------------------
# # User Input
# # -------------------------------
# if "user_query" not in st.session_state:
#     st.session_state["user_query"] = ""

# if "reset" not in st.session_state:
#     st.session_state["reset"] = False

# user_query = st.text_input(
#     "📝 Enter your question about a drug:",
#     value="" if st.session_state.reset else st.session_state.user_query,
#     key="user_query_input"
# )


# col1, col2 = st.columns([1, 1])
# submit_clicked = st.button("Ask")
# clear_clicked = st.button("Clear")

# if clear_clicked:
#     st.session_state["reset"] = True
#     st.session_state["user_query"] = ""
#     st.rerun()

# if submit_clicked:
#     if not user_query.strip():
#         st.warning("Please enter a valid question.")
#     else:
#         with st.spinner("Processing your question..."):

#             # Step 1: Query Processing
#             (intent, sub_intent), entities = preprocess_query(user_query)

#             # Step 2: Retrieval
#             retrieved_chunks = Retrieval_averagedQP(
#                 raw_query=user_query,
#                 intent=intent,
#                 entities=entities,
#                 top_k=10,
#                 alpha=0.8
#             )

#             # Step 3: Answer Generation
#             answer = answer_generation(user_query, retrieved_chunks, top_k=3)

#         # -------------------------------
#         # Chat-like Output
#         # -------------------------------
#         with st.chat_message("user"):
#             st.markdown(user_query)

#         with st.chat_message("assistant"):
#             st.markdown(f"**{answer}**")

#         with st.expander("📄 Top Retrieved Texts (Top 3 Chunks)"):
#             top_chunks = retrieved_chunks.head(3)
#             for i, row in top_chunks.iterrows():
#                 st.markdown(f"""
#                 **{i+1}. {row['drug_name']} | {row['section']} > {row['subsection']}**  
#                 {row['chunk_text']}  
#                 *Score: {round(row['semantic_similarity_score'], 3)}*
#                 """)


import streamlit as st
import pandas as pd

from Query_processing import preprocess_query
from Retrieval import Retrieval_averagedQP
from Answer_Generation import answer_generation


st.set_page_config(page_title="Drug QA Assistant")

st.write("Streamlit app started")

try:
    from Query_processing import preprocess_query
    st.write(" Query module loaded")

    from Retrieval import Retrieval_averagedQP
    st.write("Retrieval module loaded")

    from Answer_Generation import answer_generation
    st.write("Answer module loaded")
except Exception as e:
    st.error(f" App failed during import: {e}")
    raise e


# -------------------------------
# App Config and Initialization
# -------------------------------
st.set_page_config(page_title=" Drug QA Chatbot", layout="wide")
st.title("💬 Drug QA Chatbot")
st.caption("Ask any drug-related question and get reliable answers.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of dicts: {"query", "answer", "chunks"}

# -------------------------------
# Display past chat turns
# -------------------------------
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        st.markdown(f"**{turn['answer']}**")
    with st.expander(" Retrieved Context Chunks"):
        for i, row in turn["chunks"].iterrows():
            st.markdown(f"""
            **{row['drug_name']} | {row['section']} > {row['subsection']}**  
            {row['chunk_text']}  
            *Score: {round(row['semantic_similarity_score'], 3)}*
            """)
        st.markdown("---")

# -------------------------------
# User Input (chat-style)
# -------------------------------
user_query = st.chat_input("Ask a new question...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Retrieving and generating answer..."):

        # Step 1: Process query
        (intent, sub_intent), entities = preprocess_query(user_query)

        # Step 2: Retrieve relevant chunks
        retrieved_chunks = Retrieval_averagedQP(
            raw_query=user_query,
            intent=intent,
            entities=entities,
            top_k=10
        )

        # Step 3: Generate answer from top chunks
        answer = answer_generation(user_query, retrieved_chunks, top_k=3)

    # Step 4: Store and display response
    with st.chat_message("assistant"):
        st.markdown(f"**{answer}**")

    with st.expander("📄 Retrieved Context Chunks"):
        for i, row in retrieved_chunks.head(3).iterrows():
            st.markdown(f"""
            **{row['drug_name']} | {row['section']} > {row['subsection']}**  
            {row['chunk_text']}  
            *Score: {round(row['semantic_similarity_score'], 3)}*
            """)
        st.markdown("---")

    # Store this turn
    st.session_state.chat_history.append({
        "query": user_query,
        "answer": answer,
        "chunks": retrieved_chunks.head(3)
    })
