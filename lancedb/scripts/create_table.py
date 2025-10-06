import lancedb
import pyarrow as pa

# Connect to LanceDB (path is inside container, mapped to ./my_lancedb on host)
db = lancedb.connect("my_lancedb")

# # ✅ Drop the table if it exists
# if "ahsp" in db.table_names():
#     db.drop_table("ahsp")
#     print("🗑️ Table dropped: ahsp")

# # ✅ Create table with proper schema
# schema = pa.schema([
#     pa.field("name", pa.string()),
#     pa.field("description", pa.string()),
#     pa.field("code", pa.string()),
#     pa.field("classification", pa.string()),  # classification should be string
#     pa.field("vector", pa.list_(pa.float32(), list_size=768)),  # embeddings
# ])
# db.create_table("ahsp", schema=schema)

if "ahsp" not in db.table_names():
    schema = pa.schema([
        pa.field("name", pa.string()),
        pa.field("description", pa.string()),
        pa.field("code", pa.string()),
        pa.field("classification", pa.string()),  # classification should be string
        pa.field("vector", pa.list_(pa.float32(), list_size=768)),  # embeddings
    ])
    db.create_table("ahsp", schema=schema)

# ✅ Open the table
table = db.open_table("ahsp")
print("✅ Table ready:", table.name)
