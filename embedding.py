import json, os, re
from pathlib import Path
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModel

def parse_args():
    parser = argparse.ArgumentParser(description="Encode corpus into embeddings")
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Path to input JSON corpus file, e.g., data/2wikimultihopqa_corpus.json")
    parser.add_argument("--emb_path", type=Path, required=True,
                        help="Path to save embeddings .npy, e.g., embdding_all/wiki_embeddings.npy")
    parser.add_argument("--meta_path", type=Path, required=True,
                        help="Path to save meta info .jsonl, e.g., wiki_meta.jsonl")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load input data
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Load embedding model
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', trust_remote_code=True)
    model.config.use_cache = False

    max_length = 32768
    all_embeddings = []
    ids = []

    # Encode corpus
    for i, item in enumerate(raw):
        title = item.get("title", "").strip()
        text  = item.get("text", "").strip()
        content = f"{title}\n\n{text}" if title else text

        emb = model.encode(content, max_length=max_length)
        all_embeddings.append(emb)
        ids.append({"id": i, "title": title})

    # Save embeddings
    all_embeddings = np.array(all_embeddings, dtype="float32")
    print("shape:", all_embeddings.shape)
    os.makedirs(args.emb_path.parent, exist_ok=True)
    np.save(args.emb_path, all_embeddings)

    # Save metadata
    with open(args.meta_path, "w", encoding="utf-8") as f:
        for m in ids:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Verify saved embeddings
    X = np.load(args.emb_path)
    print("Loaded shape:", X.shape)

if __name__ == "__main__":
    main()
