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
db = lancedb.connect("my_lancedb")
table = db.open_table("ahsp")

st.header("🔍 VURI")

query = st.text_input(
    "What do you want to know?",
    value=st.session_state.query,
    key="query_input"
)

search_type = st.radio(
    "Search Type:",
    ["Description", "Code"],
    horizontal=True,
)

# When user clicks "Search"
if st.button("Search"):
    if not query.strip():
        st.warning("⚠️ Please enter something to search.")
    else:
        df = table.to_pandas()

        if search_type == "Description":
            # 🧠 Semantic + keyword search (same as before)
            query_vector = model.encode(query).tolist()
            vector_results = table.search(query_vector).limit(25).to_list()

            keyword_results = df[df["description"].str.contains(query, case=False, na=False)]

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
                    "code": row["code"],
                    "name": row["name"],
                    "classification": row["classification"],
                    "description": row["description"],
                    "url": row["url"],
                    "source": "keyword",
                }
                for _, row in keyword_results.iterrows()
            ])

            final_df = pd.concat([keyword_df, vector_df]).drop_duplicates(subset=["code"]).reset_index(drop=True)

        else:
            # 🔢 Partial match by code (case-insensitive)
            final_df = df[df["code"].astype(str).str.contains(query, case=False, na=False)].reset_index(drop=True)

        # 🧾 Show Results
        st.subheader("📚 Search Results")
        if final_df.empty:
            st.info("No results found.")
        else:
            for idx, row in final_df.iterrows():
                if "source" in row and row["source"] == "keyword":
                    prefix = "⭐"
                elif search_type == "Code":
                    prefix = "⭐"
                else:
                    prefix = ""
                with st.expander(f"{prefix} {idx+1}. [{row['code']}] {row['name']}"):
                    st.markdown(f"**📝 Description:** {row['description']}")
                    st.markdown(f"**🏷️ Classification:** {row['classification']}")

                    image_path = row.get("url", "")
                    if image_path:
                        if not os.path.isabs(image_path):
                            image_path = os.path.join("images", os.path.basename(image_path))
                        if os.path.exists(image_path):
                            st.image(image_path, caption=os.path.basename(image_path), use_container_width=True)
                        else:
                            st.warning(f"⚠️ Image not found at: {image_path}")
                    else:
                        st.info("No image available for this item.")
