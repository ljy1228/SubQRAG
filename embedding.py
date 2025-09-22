import json, os, re
from pathlib import Path
import numpy as np

from transformers import AutoTokenizer, AutoModel

DATA_PATH = Path("data/2wikimultihopqa_corpus.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', trust_remote_code=True)
model.config.use_cache = False


max_length = 32768
all_embeddings = []
ids = []  

for i, item in enumerate(raw):
    title = item.get("title", "").strip()
    text  = item.get("text", "").strip()
    content = f"{title}\n\n{text}" if title else text

    emb = model.encode(content,max_length = max_length)  
    all_embeddings.append(emb)
    ids.append({"id": i, "title": title})

all_embeddings = np.array(all_embeddings, dtype="float32")
print("shape:", all_embeddings.shape)  # (N, D)


np.save("embdding_all/wiki_embeddings.npy", all_embeddings)

import json
with open("wiki_meta.jsonl", "w", encoding="utf-8") as f:
    for m in ids:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

X = np.load("embdding_all/wiki_embeddings.npy")
print("Loaded shape:", X.shape)
    

    
