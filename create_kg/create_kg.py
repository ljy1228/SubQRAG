from create_kg_utils import *  
from pathlib import Path
from tqdm import tqdm
import json
import pickle
import os
from typing import Tuple, Optional


#DATA_PATH represents the path to the input JSON file containing the corpus.
#OUT_PATH is the path where the final knowledge graph will be saved.
DATA_PATH = Path("lveval_corpus.json")
OUT_PATH = Path("lveval_knowledge.pkl")
OUT_TMP = OUT_PATH.with_suffix(".pkl.tmp")
CKPT_PATH = Path("lveval_checkpoint.pkl")
CKPT_TMP = Path("lveval_checkpoint.pkl.tmp")

def atomic_dump(obj, path: Path, tmp: Path):
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

def load_corpus() -> list:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def try_save_checkpoint(processed_idx: int, union_kb: KB):
    payload = {"version": 1, "processed_idx": processed_idx}
    try:
        payload["mode"] = "kb"
        payload["kb"] = union_kb
        atomic_dump(payload, CKPT_PATH, CKPT_TMP)
        return
    except Exception:
        pass
    try:
        triples = []
        for r in union_kb.relations:
            triples.append((r.head, r.relation, r.tail))
        payload = {
            "version": 1,
            "mode": "triples",
            "processed_idx": processed_idx,
            "triples": triples,
        }
        atomic_dump(payload, CKPT_PATH, CKPT_TMP)
    except Exception as e:
        tqdm.write(f"[WARN] checkpoint fallback failed: {e}")

def try_load_checkpoint() -> Tuple[int, Optional[KB]]:
    if not CKPT_PATH.exists():
        return -1, None

    with open(CKPT_PATH, "rb") as f:
        payload = pickle.load(f)

    processed_idx = payload.get("processed_idx", -1)
    mode = payload.get("mode", "kb")

    if mode == "kb" and "kb" in payload:
        return processed_idx, payload["kb"]

    if mode == "triples" and "triples" in payload:
        kb = KB()
        for h, rel, t in payload["triples"]:
            kb.add_relation(Relation(h, rel, t))
        return processed_idx, kb
    return processed_idx, None

def save_snapshot(union_kb: KB):
    nodes, rel_index, triple_index = build_indices(union_kb.relations)
    atomic_dump((nodes, rel_index, triple_index), OUT_TMP, OUT_TMP)  
    with open(OUT_TMP, "wb") as f:
        pickle.dump((nodes, rel_index, triple_index), f)
    os.replace(OUT_TMP, OUT_PATH)

def main():
    corpus = load_corpus()  
    processed_idx, union_kb = try_load_checkpoint()
    if union_kb is None:
        union_kb = KB()
    start_idx = processed_idx + 1
    if start_idx < 0:
        start_idx = 0

    pbar = tqdm(range(start_idx, len(corpus)), desc="Building KB (resume)")
    for i in pbar:
        try:
            article = corpus[i]
            text = article.get("text", "")
            kb_piece = from_small_text_to_kb(text, verbose=False)
            for r in kb_piece.relations:
                union_kb.add_relation(r)
            nodes, rel_index, triple_index = build_indices(union_kb.relations)
            with open(OUT_TMP, "wb") as f:
                pickle.dump((nodes, rel_index, triple_index), f)
            os.replace(OUT_TMP, OUT_PATH)
            try_save_checkpoint(processed_idx=i, union_kb=union_kb)

        except Exception as e:
            tqdm.write(f"[WARN] i={i} failed: {e}")
            try_save_checkpoint(processed_idx=i - 1, union_kb=union_kb)
            continue

    if OUT_PATH.exists():
        with open(OUT_PATH, "rb") as f:
            entities_loaded, attributes_loaded, triples_loaded = pickle.load(f)
        print("Loaded knowledge base snapshot:")
        print("Entities count:", len(entities_loaded))
        print("Attributes count:", len(attributes_loaded))
        print("Triples count:", triples_loaded)


if __name__ == "__main__":
    main()
