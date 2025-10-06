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

    # ✅ Generate embeddings dari kolom "description"
    df["vector"] = df["description"].apply(lambda x: model.encode(x).tolist())

    # ✅ Preview data yang akan diinsert
    st.write("📋 Preview parsed data:", df[["name", "description", "code", "classification"]].head())

    # ✅ Insert all
    if st.button("Insert All into LanceDB"):
        records = df[["name", "description", "code", "classification", "vector"]].to_dict(orient="records")
        table.add(records)
        st.success(f"✅ Inserted {len(records)} records into LanceDB")
