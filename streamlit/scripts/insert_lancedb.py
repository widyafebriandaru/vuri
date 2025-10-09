import streamlit as st
import lancedb
import pandas as pd # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import re

# Connect to LanceDB
db = lancedb.connect("my_lancedb")

# ✅ Always open the table (assume it was created beforehand)
table = db.open_table("ahsp")

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2")

st.title("LanceDB AHSP Manager")

# ======================
# Section: Insert from Excel
# ======================
st.header("Insert from Excel")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None)  # no header, column A = col 0
    df = df.dropna()

    # ✅ Preprocess description (lowercase, strip spaces, normalize)
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"\s+", " ", text)   # remove multiple spaces
        return text.strip()

    # ✅ Mapping sesuai struktur Excel
    df["name"] = df[0]               # column A
    df["description"] = df[1].apply(clean_text)  # column B
    df["code"] = df[2]               # column C
    df["classification"] = df[3]     # column D
    df["url"] = df[4] # column E

    # ✅ Generate embeddings dari kolom "description"
    df["vector"] = df["description"].apply(lambda x: model.encode(x).tolist())

    # ✅ Preview data yang akan diinsert
    st.write("📋 Preview parsed data:", df[["name", "description", "code", "classification", "url"]].head())

    # ✅ Insert all
if st.button("Insert All into LanceDB"):
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Automatically add "images/" prefix to the 'url' column
    df_copy["url"] = df_copy["url"].apply(lambda x: f"images/{x}" if not str(x).startswith("images/") else x)

    # Convert to list of dicts for LanceDB
    records = df_copy[["name", "description", "code", "classification", "vector", "url"]].to_dict(orient="records")

    # Insert into LanceDB
    table.add(records)

    st.success(f"✅ Inserted {len(records)} records into LanceDB")