import os
import re
import json
import time
from collections import defaultdict, deque
from pathlib import Path
import networkx as nx
import spacy
import matplotlib.pyplot as plt
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from kg import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SubQ-RAG: decompose → retrieve → update → generate (parameterized paths)")
    # inputs
    parser.add_argument("--decomp_path", type=Path, default=Path("decompose_data/decompose_2wikimultihopqa.jsonl"),
                        help="Path to decomposed sub-questions (.jsonl).")
    parser.add_argument("--corpus_path", type=Path, default=Path("SubQRAG/data/2wikimultihopqa_corpus.json"),
                        help="Path to corpus JSON.")
    parser.add_argument("--emb_path", type=Path, default=Path("embdding_all/wiki_embeddings.npy"),
                        help="Path to precomputed article embeddings (.npy).")
    parser.add_argument("--sent_model_path", type=str, default="/mnt/ljy/lag_better/all-MiniLM-L6-v2",
                        help="SentenceTransformer model path or name.")

    # knowledge graph files
    parser.add_argument("--kg_pickle_in", type=Path, default=Path("/mnt/ljy/lag_better/kg_dataset/dynamic_kb.pkl"),
                        help="Existing KB pickle to load (input).")
    parser.add_argument("--kg_pickle_out", type=Path, default=Path("kg_dataset/2wikimultihopqa_dynamic_kb.pkl"),
                        help="Where to save updated dynamic KB pickle (output).")

    # outputs
    parser.add_argument("--out_path", type=Path, default=Path("result/2wiki/2wiki_update_graph.jsonl"),
                        help="Path to save final results (.jsonl).")

    # OpenAI client configs 
    parser.add_argument("--openai_base_url", type=str, default="",
                        help="OpenAI-compatible base URL (if using a proxy/compatible server).")
    parser.add_argument("--openai_api_key", type=str, default="",
                        help="OpenAI API key. If empty, will try env var OPENAI_API_KEY.")

    return parser.parse_args()

def main():
    args = parse_args()

    # -------- OpenAI client --------
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(
        base_url=args.openai_base_url,
        api_key=api_key,
    )

    # -------- constants/keywords --------
    bad_keywords = [
        "not provided", "facts", "no answer", "unknown", "n/a", "missing", "null", "undefined"
    ]

    placeholder_pat = re.compile(r"#(\d+)")

    def resolve_placeholders(text, answers):
        def repl(m):
            k = int(m.group(1)) - 1
            return answers[k] if 0 <= k < len(answers) and answers[k] else m.group(0)
        return placeholder_pat.sub(repl, text)

    def extract_rewrite_tag(text):
        m = re.search(r"<rewrite>(.*?)</rewrite>", text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else text.strip()

    with open(args.corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    sentence_model = SentenceTransformer(args.sent_model_path)

    data = []
    with open(args.decomp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    X = np.load(args.emb_path)

    kb = build_kb_from_pickle(str(args.kg_pickle_in), out_path=str(args.kg_pickle_out))


    for i, item in enumerate(data, 1):
        decomposition = item.get("decomposition", "")
        matches = re.findall(r'(\d+)\.\s*(.*?)(?=\n\d+\.|$)', decomposition, flags=re.S)
        questions = [q.strip() for _, q in matches]  # decomposition sub-questions
        print("all_subqueation", questions)

        response = ""
        answers = []
        facts = []

        for idx, original_question in enumerate(questions):
            if idx == 0:
                question = original_question
            else:
                question = rewrite_with_llm(original_question, answers[:idx], client)  # rewrite

            nodes, rel_index, triple_index = build_indices([
                {"head": r.head, "type": r.relation, "tail": r.tail}
                for r in kb.relations
            ])

            support = select_support_triples(question, kb.relations, sentence_model, topk=5)
            print("support_5", idx, support)

            response, fact = answer_with_triples_llm_simple(question, support, client)

            if any(kw in response.lower() for kw in bad_keywords):
                print("error")
                help_article = ''
                question_embedding = sentence_model.encode(question)
                similarities = sentence_model.similarity(question_embedding, X).tolist()
                top3_idx = np.argsort(similarities[0])[-3:][::-1]

                for top_item in top3_idx:
                    help_article = help_article + corpus[top_item].get("text", "")

                response, fact = answer_with_article_triple(help_article, question, client)
                if not any(kw in response.lower() for kw in bad_keywords):
                    print("corrected")
                    update_triples = from_small_text_to_triples(help_article, client)
                    added = kb.add_relations_from_triples(update_triples, provenance="hotpotqa_article_42")
                    print(fact)
                    kb.save(str(args.kg_pickle_out))  

            facts.append(fact)
            answers.append(response)

            print("response", response)

        final_question = item.get("question", "")
        final_response = answer_final_withlogic(final_question, facts, client)
        item["response"] = final_response
        print("final_response", final_response)

        os.makedirs(args.out_path.parent, exist_ok=True)
        with open(args.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
