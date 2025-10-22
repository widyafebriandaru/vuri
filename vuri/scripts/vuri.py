import streamlit as st
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
import os

# ======================
# Session State & Model
# ======================
if "query" not in st.session_state:
    st.session_state.query = ""
if "ai_responses" not in st.session_state:
    st.session_state.ai_responses = {}
if "expanded_states" not in st.session_state:
    st.session_state.expanded_states = {}

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model_safely():
    return SentenceTransformer("all-mpnet-base-v2", device="cpu", trust_remote_code=True)

# Connect LanceDB
model = load_model_safely()
db = lancedb.connect("my_lancedb")
table = db.open_table("ahsp")

st.header("🔍 VURI")

# ======================
# Input Fields
# ======================
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

# ======================
# Search Logic
# ======================
if st.button("Search", key="search_button"):
    if not query.strip():
        st.warning("⚠️ Please enter something to search.")
    else:
        df = table.to_pandas()

        if search_type == "Description":
            # 🔹 Vector Search
            query_vector = model.encode(query).tolist()
            vector_results = table.search(query_vector).limit(25).to_list()

            # 🔹 Keyword Search
            keyword_results = df[df["description"].str.contains(query, case=False, na=False)]

            # Convert vector + keyword results to DataFrame
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
            final_df = df[df["code"].astype(str).str.contains(query, case=False, na=False)].reset_index(drop=True)

        # Store results in session state
        st.session_state.search_results = final_df
        st.session_state.show_results = True

# ======================
# Display Results
# ======================
if st.session_state.get("show_results", False) and "search_results" in st.session_state:
    final_df = st.session_state.search_results
    
    st.subheader("📚 Search Results")
    if final_df.empty:
        st.info("No results found.")
    else:
        for idx, row in final_df.iterrows():
            source = row.get("source", "")
            prefix = "⭐" if source == "keyword" or search_type == "Code" else ""

            # Create unique keys
            item_key = f"{row['code']}_{idx}"
            expander_key = f"expander_{item_key}"
            form_key = f"form_{item_key}"
            
            # Initialize expander state
            if expander_key not in st.session_state.expanded_states:
                st.session_state.expanded_states[expander_key] = True

            # Use form for each item
            with st.form(key=form_key):
                # Display content in expander
                with st.expander(f"{prefix} {idx+1}. [{row['code']}] {row['name']}", 
                               expanded=st.session_state.expanded_states[expander_key]):
                    
                    st.markdown(f"**📝 Description:** {row['description']}")
                    st.markdown(f"**🏷️ Classification:** {row['classification']}")

                    # 🖼️ Image Display
                    img_name = os.path.basename(row.get("url", ""))
                    image_path = os.path.join("images", img_name) if img_name else ""
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, caption=img_name, use_container_width=True)
                    elif image_path:
                        st.warning(f"⚠️ Image not found at: {image_path}")
                    else:
                        st.info("No image available for this item.")

                # AI Button outside expander but inside form
                ai_key = f"ai_{row['code']}"
                ai_clicked = st.form_submit_button(f"🤖 AI Explain [{row['code']}]")
                
                if ai_clicked:
                    # Keep expander open
                    st.session_state.expanded_states[expander_key] = True
                    
                    API_KEY = os.getenv("MISTRAL_API_KEY", "lvvzcfsbq8P3LR4E8F7fTR9LL7GpJUG3")
                    headers = {
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    }

                    prompt = f"""
Jelaskan item berikut dalam Bahasa Indonesia dengan cara yang mudah dimengerti oleh pengguna awam:

Deskripsi: {row['description']}
"""

                    payload = {
                        "model": "mistral-tiny",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that explains construction items in Bahasa Indonesia."},
                            {"role": "user", "content": prompt},
                        ],
                    }

                    with st.spinner("🤖 Generating AI explanation..."):
                        try:
                            response = requests.post(
                                "https://api.mistral.ai/v1/chat/completions",
                                json=payload,
                                headers=headers,
                                timeout=30
                            )
                            response.raise_for_status()
                            answer = response.json()["choices"][0]["message"]["content"]
                            st.session_state.ai_responses[ai_key] = answer
                            st.rerun()  # Refresh to show the AI response
                        except Exception as e:
                            st.error(f"Error: {e}")

            # Display AI response (outside the form)
            if ai_key in st.session_state.ai_responses:
                st.markdown("### 💬 AI Explanation")
                st.write(st.session_state.ai_responses[ai_key])
                st.markdown("---")