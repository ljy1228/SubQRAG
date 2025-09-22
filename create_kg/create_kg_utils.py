
import json
import re
from collections import defaultdict
from openai import OpenAI
import time
import time
import re


DEVICE = "cuda:1"
client = OpenAI(api_key="")

class KB():
    def __init__(self):
        self.relations = []

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")


class Relation:
    def __init__(self, head: str, relation: str, tail: str,
                 confidence: float = 1.0, provenance: str = "", timestamp: float = None):
        self.head = head.strip()
        self.relation = relation.strip()
        self.tail = tail.strip()
        self.confidence = confidence
        self.provenance = provenance
        self.timestamp = timestamp or time.time()

    def __repr__(self):
        return f"({self.head}, {self.relation}, {self.tail}, conf={self.confidence:.2f})"


def build_indices(relations):
    """
    relations: List[{'head':..., 'type':..., 'tail':...}]
    Returns:
    nodes: all node names (set)
    rel_index: dict[str -> List[str]]     # candidate relation list for each head
    triple_index: dict[(head, rel) -> List[str]]  # mapping (head, rel) -> tails
    """

    nodes = set()
    rel_index = defaultdict(set)
    triple_index = defaultdict(list)
    for r in relations:
        h = (r.get('head') or '').strip()
        t = (r.get('tail') or '').strip()
        rel = (r.get('type') or '').strip()
        if not h or not rel or not t:
            continue
        nodes.add(h); nodes.add(t)
        rel_index[h].add(rel)
        triple_index[(h, rel)].append(t)

    rel_index = {k: sorted(list(v)) for k, v in rel_index.items()}
    return nodes, rel_index, triple_index

def triples_to_strings(relations):
    fact_strs = []
    for r in relations:
        if isinstance(r, dict):
            s, p, o = r.get("head"), r.get("type") or r.get("relation"), r.get("tail")
        elif isinstance(r, tuple) and len(r) == 3:
            s, p, o = r
        elif hasattr(r, "head") and hasattr(r, "relation") and hasattr(r, "tail"):
            s, p, o = r.head, r.relation, r.tail
        else:
            continue

        fact_strs.append(f"({s}, {p}, {o})")
    return fact_strs
    

def from_small_text_to_kb(text, verbose=False):
    kb = KB()

    prompt = f"""
    Please extract knowledge triples from the following text. 
    Return ONLY a valid JSON array (no extra words, no code fences).
    Each element must be in the form: {{"subject": "", "predicate": "", "object": ""}}
    Text: {text}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()

    cleaned = content.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\[.*\]", cleaned, re.S)
    if match:
        cleaned = match.group(0)
    triples = []
    try:
        triples = json.loads(cleaned)
    except json.JSONDecodeError as e:
        if verbose:
            print("Json failed:", e)
            print("text:", content)
    for triple in triples:
        if isinstance(triple, dict) and "subject" in triple and "predicate" in triple and "object" in triple:
            kb.add_relation({
                "head": triple["subject"].strip(),
                "type": triple["predicate"].strip(),
                "tail": triple["object"].strip()
            })
    

    return kb






