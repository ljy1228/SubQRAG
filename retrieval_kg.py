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
in_path = "decompose_data/decompose_2wikimultihopqa.jsonl"

kg_path = 'kg_dataset/2wikimultihopqa_knowledge.pkl'
DATA_PATH = Path("SubQRAG/data/2wikimultihopqa_corpus.json")

client = OpenAI(
  base_url="",
  api_key="",
)

with open(kg_path, "rb") as f:
    entities_loaded, attributes_loaded, triples_loaded = pickle.load(f)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)
    
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
data = []
with open(in_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data.append(json.loads(line))


X = np.load("embdding_all/wiki_embeddings.npy")
out_path = "result/wiki/wiki.jsonl"
placeholder_pat = re.compile(r"#(\d+)")

def resolve_placeholders(text, answers):
    def repl(m):
        k = int(m.group(1)) - 1
        return answers[k] if 0 <= k < len(answers) and answers[k] else m.group(0)
    return placeholder_pat.sub(repl, text)

def extract_rewrite_tag(text):
    m = re.search(r"<rewrite>(.*?)</rewrite>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()

kb = KnowledgeBase()
for (subject, predicate), obj in triples_loaded.items():
    if isinstance(obj, str):
        tail = obj.strip()
    elif isinstance(obj, list):
        tail = ", ".join([str(o).strip() for o in obj])
    else:
        tail = str(obj).strip()
    kb.add_relation({
        "head": subject.strip(),
        "type": predicate.strip(),
        "tail": tail
    })


for i, item in enumerate(data, 1):
    decomposition = item.get("decomposition", "")
    matches = re.findall(r'(\d+)\.\s*(.*?)(?=\n\d+\.|$)', decomposition, flags=re.S)  # extracted decomposed questions
    questions = [q.strip() for _, q in matches]  # get sub-questions
    print("all_subqueation",questions)
    retrival_article = []   # retrieved related articles
    
    response = ""   # answer
    answers = []    # answers
    
    facts = []
    for idx, original_question in enumerate(questions):  # iterate over each sub-question
        if idx == 0:
            question = original_question  # the first round is the original question, no need to rewrite like #1, #2
        else:
            question = rewrite_with_llm(original_question, answers[:idx], client)  # rewrite the question
                
        nodes, rel_index, triple_index = build_indices(kb.relations)
        
        
        support = select_support_triples(question, kb.relations, sentence_model, topk=5)
        
        print("support_5",idx,support)
        response,fact = answer_with_triples_llm_simple(question, support, client)
        
        print("response",response)
        print("fact",fact)
        if "not provided" in response.lower() or "facts" in response.lower() :
            help_article = ''
            question_embedding = sentence_model.encode(question)  # encode the question into embedding
            similarities = sentence_model.similarity(question_embedding, X).tolist()  # compute similarity between query and article embeddings
            top3_idx = np.argsort(similarities[0])[-3:][::-1]  # select the top 3 articles, get their ids
            for top_item in top3_idx:
                help_article = help_article + corpus[top_item].get("text", "")
                response = answer_with_article(help_article, question, client)
        facts.append(fact)
        answers.append(response)
        print("question",question)
        print("response",response)

    final_question = item.get("question", "")

    final_response = answer_final_withlogic(final_question, facts, client)

    item["response"] = final_response

    print("final_response",final_response)

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
