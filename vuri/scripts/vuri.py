import streamlit as st
import lancedb
import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
import os

# ======================
# Session State & Model
# ======================
if "query" not in st.session_state:
    st.session_state.query = ""

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model_safely():
    return SentenceTransformer("all-mpnet-base-v2", device="cpu", trust_remote_code=True)

model = load_model_safely()

# ======================
# Connect to LanceDB
# ======================
db = lancedb.connect("my_lancedb")
table = db.open_table("ahsp")

# ======================
# Streamlit UI
# ======================
st.header("🔍 VURI")

with st.form(key="query_form"):
    query = st.text_input(
        "What do you want to know?",
        value=st.session_state.query,
        key="query_input"
    )
    submitted = st.form_submit_button("Ask")

# ======================
# Search Logic
# ======================
if submitted and query:
    st.session_state.query = ""  # reset input

    # 1️⃣ Vector Search
    query_vector = model.encode(query).tolist()
    vector_results = table.search(query_vector).limit(25).to_list()

    # 2️⃣ Exact Keyword Match (case-insensitive)
    keyword_results = table.to_pandas()
    keyword_results = keyword_results[
        keyword_results["description"].str.contains(query, case=False, na=False)
    ]

    # 3️⃣ Convert both to DataFrame
    vector_df = pd.DataFrame([
        {
            "code": item.get("code", ""),
            "name": item.get("name", ""),
            "classification": item.get("classification", ""),
            "description": item.get("description", ""),
            "url": item.get("url", ""),
            "source": "vector",
        }
        for item in vector_results
    ])

    keyword_df = pd.DataFrame([
        {
            "code": row.get("code", ""),
            "name": row.get("name", ""),
            "classification": row.get("classification", ""),
            "description": row.get("description", ""),
            "url": row.get("url", ""),
            "source": "keyword",
        }
        for _, row in keyword_results.iterrows()
    ])

    # 4️⃣ Merge results — prioritize keyword match first
    final_df = pd.concat([keyword_df, vector_df]).drop_duplicates(subset=["code"]).reset_index(drop=True)

    st.subheader("📚 Hasil Pencarian")

    if final_df.empty:
        st.info("Tidak ada hasil ditemukan.")
    else:
        for idx, row in final_df.iterrows():
            prefix = "⭐" if row["source"] == "keyword" else ""
            with st.expander(f"{prefix} {idx+1}. [{row['code']}] {row['name']}"):
                st.markdown(f"**📝 Deskripsi:** {row['description']}")
                st.markdown(f"**🏷️ Klasifikasi:** {row['classification']}")

                # 🖼️ Image Display & Download
                image_path = row.get("url", "")
                if image_path:
                    # If path is relative, assume inside ./images/
                    if not os.path.isabs(image_path):
                        image_path = os.path.join("images", os.path.basename(image_path))

                    if os.path.exists(image_path):
                        st.image(image_path, caption=os.path.basename(image_path), use_container_width=True)

                        with open(image_path, "rb") as file:
                            st.download_button(
                                label="⬇️ Download Image",
                                data=file,
                                file_name=os.path.basename(image_path),
                                mime="image/png"
                            )
                    else:
                        st.warning(f"⚠️ Image not found at: {image_path}")
                else:
                    st.info("No image available for this item.")
