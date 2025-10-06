import streamlit as st
import lancedb
import pandas as pd # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore

# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ""

# Load embedding model once
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model_safely():
    return SentenceTransformer("all-mpnet-base-v2", device="cpu", trust_remote_code=True)

model = load_model_safely()

# Connect to LanceDB
db = lancedb.connect("my_lancedb")
table = db.open_table("ahsp")

st.header("🔍 VURI")

with st.form(key='query_form'):
    query = st.text_input("What do you want to know?", value=st.session_state.query, key="query_input")
    submitted = st.form_submit_button("Ask")

    if submitted and query:
        st.session_state.query = ""  # reset input

        # 🔹 1) Vector search
        query_vector = model.encode(query).tolist()
        vector_results = table.search(query_vector).limit(25).to_list()

        # 🔹 2) Exact keyword match (case-insensitive)
        keyword_results = table.to_pandas()
        keyword_results = keyword_results[
            keyword_results["description"].str.contains(query, case=False, na=False)
        ]

        # 🔹 Convert keduanya ke DataFrame dengan struktur sama
        vector_df = pd.DataFrame([
            {
                "code": item.get("code", ""),
                "name": item.get("name", ""),
                "classification": item.get("classification", ""),
                "description": item.get("description", ""),
                "source": "vector"
            }
            for item in vector_results
        ])

        keyword_df = pd.DataFrame([
            {
                "code": row.get("code", ""),
                "name": row.get("name", ""),
                "classification": row.get("classification", ""),
                "description": row.get("description", ""),
                "source": "keyword"
            }
            for _, row in keyword_results.iterrows()
        ])

        # 🔹 Gabungkan: utamakan keyword match dulu
        final_df = pd.concat([keyword_df, vector_df]).drop_duplicates(subset=["code"]).reset_index(drop=True)

        st.subheader("📚 Hasil Pencarian")
        if final_df.empty:
            st.info("Tidak ada hasil ditemukan.")
        else:
            for idx, row in final_df.iterrows():
                prefix = "⭐" if row["source"] == "keyword" else " "
                st.markdown(
                    f"""
                    {prefix} **{idx+1}. [{row['code']}] {row['name']}**
                    - 🏷️ *Klasifikasi:* {row['classification']}
                    - 📝 *Deskripsi:* {row['description']}
                    """
                )