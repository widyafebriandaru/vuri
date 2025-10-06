import streamlit as st
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Initialize session state variables
if 'new_note' not in st.session_state:
    st.session_state.new_note = ""
if 'query' not in st.session_state:
    st.session_state.query = ""

# Load embedding model once
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model_safely():
    return SentenceTransformer("all-mpnet-base-v2", device="cpu", trust_remote_code=True)

# Load model
model = load_model_safely()

# Connect to LanceDB
db = lancedb.connect("my_lancedb")
table = db.open_table("ahsp")

# ✅ Main Panel: Ask Assistant
st.header("🔍 VURI")

# Use a form to clear the input after submission
with st.form(key='query_form'):
    query = st.text_input("What do you want to know?", value=st.session_state.query, key="query_input")
    submitted = st.form_submit_button("Ask")
    
    if submitted and query:
        st.session_state.query = ""  # Clear the query input
        query_vector = model.encode(query).tolist()
        
        # Perform search and get results
        results = table.search(query_vector).limit(25).to_list()
        
        # Convert to DataFrame dengan hasil
        results_df = pd.DataFrame([
            {
                'code': item.get('code', ''), 
                'description': item['description']
            } 
            for item in results
        ])
        
        st.subheader("📚 Hasil pencarian (Top 25)")
        for idx, row in results_df.iterrows():
            st.markdown(f"**{idx+1}. [{row['code']}] {row['description']}**")
