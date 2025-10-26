import streamlit as st
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
import os
import numpy as np
from difflib import SequenceMatcher
import re

# ======================
# Session State & Model
# ======================
st.set_page_config(
    page_title="VURI",
    initial_sidebar_state="collapsed"
)

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

st.header("(VURI) Verifikasi item Untuk RAB Inpres")

# ======================
# Input Fields
# ======================
query = st.text_input(
    "Apa yang ingin dicari?",
    value=st.session_state.query,
    key="query_input",
    placeholder="Bongkar pasangan batu"
)

search_type = st.radio(
    "Cari berdasarkan:",
    ["Deskripsi", "Kode"],
    horizontal=True,
)

# ======================
# Search Logic
# ======================
############################ Sidebar ###############################
st.sidebar.header("⚙️ Pengaturan Pencarian")

st.sidebar.markdown("### 📚 Filter Regulasi")
filter_se182 = st.sidebar.toggle("SE 182", value=False)
filter_se30 = st.sidebar.toggle("SE 30", value=False)
filter_pu8 = st.sidebar.toggle("Permen PU 8", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Manajemen Data")
st.sidebar.write("Update Database:")
st.sidebar.link_button("📥 Buka Aplikasi Input Data", "http://10.123.1.200:8501/", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Navigasi Cepat")
st.sidebar.markdown("""
- [📊 Dashboard Monitoring](http://10.123.1.200:3000/)
- [🧠 Panduan Penggunaan](http://10.123.1.200/docs)
- [💾 Backup Database](http://10.123.1.200:5050/)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Tentang Aplikasi")
st.sidebar.info("""
**VURI (Verifikasi Item untuk RAB Inpres)**  
Dibuat untuk membantu pencarian, validasi, dan penjelasan item pekerjaan konstruksi berbasis semantik.
Versi: 1.2.5  
Dikembangkan oleh: *Widya Febriandaru*
""")

with st.sidebar.expander("💬 Kirim Masukan"):
    feedback = st.text_area("Tulis komentar atau saran:")
    if st.button("Kirim", key="feedback_btn"):
        st.success("Terima kasih atas masukannya!")
############################ Sidebar ###############################

if st.button("Cari", key="search_button"):
    if not query.strip():
        st.warning("⚠️ Please enter something to search.")
    else:
        df = table.to_pandas()

        # 🟩 MOVE FILTER LOGIC HERE - Define active_filters at the beginning
        active_filters = []
        if filter_se182:
            active_filters.append("Bidang PSDA - SE no.182 Tahun 2025")
        if filter_se30:
            active_filters.append("Bidang Bina Marga - SE no.182 Tahun 2025")
        if filter_pu8:
            active_filters.append("Permen PU no.8 Tahun 2023")

        if search_type == "Deskripsi":
            query_vector = model.encode(query).tolist()
            vector_results = table.search(query_vector).limit(100).to_list()
            df = table.to_pandas()

            # Convert vector search results into a DataFrame
            vector_df = pd.DataFrame([
                {
                    "code": item.get("code", ""),
                    "name": item.get("name", ""),
                    "classification": item.get("classification", ""),
                    "description": item.get("description", ""),
                    "url": item.get("url", ""),
                    # Convert distance to similarity (1 - normalized distance)
                    "semantic_score": max(0.0, 1 - item.get("_distance", 1.0)),
                }
                for item in vector_results
            ])

            # Tokenize query for lexical comparison
            query_tokens = re.findall(r"\w+", query.lower())

            # === Keyword Overlap Function ===
            def keyword_overlap(desc):
                desc_tokens = re.findall(r"\w+", str(desc).lower())
                if not query_tokens or not desc_tokens:
                    return 0
                matches = sum(token in desc_tokens for token in query_tokens)
                return matches / len(query_tokens)

            # === Fuzzy Ratio Function (handles word order / typo / inflection) ===
            def fuzzy_ratio(a, b):
                return SequenceMatcher(None, a.lower(), b.lower()).ratio()

            # Apply both scoring functions
            vector_df["keyword_score"] = vector_df["description"].apply(keyword_overlap)
            vector_df["fuzzy_score"] = vector_df["description"].apply(lambda x: fuzzy_ratio(query, str(x)))

            # Normalize each score to 0–1 range
            for col in ["semantic_score", "keyword_score", "fuzzy_score"]:
                if vector_df[col].max() > 0:
                    vector_df[col] = vector_df[col] / vector_df[col].max()

            # === Hybrid Weighted Scoring ===
            # α = lexical priority, β = fuzzy, γ = semantic
            alpha = 0.55   # keyword importance
            beta = 0.25    # fuzzy importance
            gamma = 0.20   # semantic importance

            vector_df["hybrid_score"] = (
                alpha * vector_df["keyword_score"]
                + beta * vector_df["fuzzy_score"]
                + gamma * vector_df["semantic_score"]
            )

            # Sort by hybrid score (descending)
            final_df = vector_df.sort_values(by="hybrid_score", ascending=False).head(15).reset_index(drop=True)
            
            # 🟩 APPLY FILTERS AFTER final_df IS DEFINED
            if active_filters:
                final_df = final_df[final_df["classification"].isin(active_filters)]
            
        else:
            # Code-based search
            final_df = df[df["code"].astype(str).str.contains(query, case=False, na=False)].reset_index(drop=True)
            
            # 🟩 APPLY FILTERS TO CODE-BASED SEARCH TOO
            if active_filters:
                final_df = final_df[final_df["classification"].isin(active_filters)]
            
            # Limit results to 15 items
            final_df = final_df.head(15)
        
        # Store results in session state
        st.session_state.search_results = final_df
        st.session_state.show_results = True
        st.session_state.active_filters = active_filters  # Store for display

# ======================
# Display Results
# ======================
if st.session_state.get("show_results", False) and "search_results" in st.session_state:
    final_df = st.session_state.search_results
    active_filters = st.session_state.get("active_filters", [])
    
    st.subheader("📚 Hasil pencarian")
    if active_filters:
        st.info(f"Filter aktif: {', '.join(active_filters)}")

    st.info(f"Menampilkan {len(final_df)} buah hasil pencarian")
    
    if final_df.empty:
        st.info("Tidak ada hasil yang cocok.")
    else:
        for idx, row in final_df.iterrows():
            source = row.get("source", "")
            prefix = "⭐" if source == "keyword" or search_type == "Kode" else ""

            # Create unique keys
            item_key = f"{row['code']}_{idx}"
            expander_key = f"expander_{item_key}"
            ai_key = f"ai_{row['code']}"
            
            # Initialize expander state as CLOSED
            if expander_key not in st.session_state.expanded_states:
                st.session_state.expanded_states[expander_key] = False

            # Display content in expander - initially CLOSED
            with st.expander(f"{prefix} {idx+1}. [{row['code']}] {row['name']}", 
                           expanded=st.session_state.expanded_states[expander_key]):
                
                st.markdown(f"**📝 Deskripsi:** {row['description']}")
                st.markdown(f"**🏷️ Klasifikasi:** {row['classification']}")

                # 🖼️ Image Display
                img_name = os.path.basename(row.get("url", ""))
                image_path = os.path.join("images", img_name) if img_name else ""
                if image_path and os.path.exists(image_path):
                    st.image(image_path, caption=img_name, use_container_width=True)
                elif image_path:
                    st.warning(f"⚠️ Gambar tidak tersedia: {image_path}")
                else:
                    st.info("No image available for this item.")

                # 🤖 AI Explain Button INSIDE the expander
                ai_clicked = st.button(f"🔍Jelaskan [{row['code']}]", key=f"ai_btn_{item_key}")
                
                if ai_clicked:
                    # Keep expander open after AI click
                    st.session_state.expanded_states[expander_key] = True
                    
                    API_KEY = os.getenv("MISTRAL_API_KEY", "lvvzcfsbq8P3LR4E8F7fTR9LL7GpJUG3")
                    headers = {
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    }

                    prompt = f"""
Anda adalah asisten teknis bidang konstruksi yang bekerja untuk Direktorat Jenderal Sumber Daya Air, Kementerian Pekerjaan Umum dan Perumahan Rakyat (PUPR). 
Tugas Anda adalah menjelaskan uraian pekerjaan konstruksi dengan bahasa yang sederhana dan kontekstual di lingkungan Pengelolaan Sumber Daya Air (PSDA).

Berikan penjelasan dalam Bahasa Indonesia yang:
- Menjelaskan apa arti pekerjaan tersebut dan bagaimana pelaksanaannya secara umum di lapangan,
- Menyebutkan alat atau metode yang biasa digunakan (jika relevan),
- Menyebutkan tujuan pekerjaan tersebut dalam konteks irigasi, sungai, bendung, embung, atau infrastruktur air lainnya,
- Menggunakan bahasa yang mudah dipahami oleh staf lapangan atau pengawas pekerjaan, bukan akademis.

Berikut item yang harus dijelaskan:

Deskripsi: {row['description']}
"""

                    payload = {
                        "model": "mistral-tiny",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that explains construction items in Bahasa Indonesia."},
                            {"role": "user", "content": prompt},
                        ],
                    }

                    with st.spinner("Harap tunggu..."):
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

                # Display AI response INSIDE the expander (below the AI button)
                if ai_key in st.session_state.ai_responses:
                    st.markdown("### 💬 Penjelasan item")
                    st.write(st.session_state.ai_responses[ai_key])