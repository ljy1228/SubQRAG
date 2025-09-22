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

kb = build_kb_from_pickle("kg_path",
                          out_path="/mnt/ljy/lag_better/kg_dataset/dynamic_kb.pkl") 


DATA_PATH = Path("SubQRAG/data/2wikimultihopqa_corpus.json")
kg_path_update = "kg_dataset/2wikimultihopqa_dynamic_kb.pkl"
client = OpenAI(
  base_url="",
  api_key="",
)

bad_keywords = [
    "not provided",
    "facts",
    "no answer",
    "unknown",
    "n/a",
    "missing",
    "null",
    "undefined"
]



with open(DATA_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)
    
sentence_model = SentenceTransformer('/mnt/ljy/lag_better/all-MiniLM-L6-v2')
data = []
with open(in_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data.append(json.loads(line))


X = np.load("embdding_all/wiki_embeddings.npy")
out_path = "result/2wiki/2wiki_update_graph.jsonl"
placeholder_pat = re.compile(r"#(\d+)")

def resolve_placeholders(text, answers):
    def repl(m):
        k = int(m.group(1)) - 1
        return answers[k] if 0 <= k < len(answers) and answers[k] else m.group(0)
    return placeholder_pat.sub(repl, text)

def extract_rewrite_tag(text):
    m = re.search(r"<rewrite>(.*?)</rewrite>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


kb = build_kb_from_pickle("/mnt/ljy/lag_better/kg_dataset/dynamic_kb.pkl",
                          out_path="/mnt/ljy/lag_better/kg_dataset/dynamic_kb.pkl")   #kg


for i, item in enumerate(data, 1):
    decomposition = item.get("decomposition", "")
    matches = re.findall(r'(\d+)\.\s*(.*?)(?=\n\d+\.|$)', decomposition, flags=re.S)
    questions = [q.strip() for _, q in matches]  #decomposition sub-questions
    print("all_subqueation",questions)
    retrival_article = []   #relevant article
    
    response = ""   
    answers = []    #answer
    
    facts = []
    for idx, original_question in enumerate(questions):  #Iterate over each sub-question
        if idx == 0:
            question = original_question  #The first round is the original question, no need to rewrite it with #1, #2, etc.
        else:
            question = rewrite_with_llm(original_question, answers[:idx], client)  #rewrite
            
        nodes, rel_index, triple_index = build_indices([
                    {"head": r.head, "type": r.relation, "tail": r.tail}
                    for r in kb.relations
                ])  
        support = select_support_triples(question, kb.relations, sentence_model, topk=10) 
        print("support_10",idx,support)
        response,fact = answer_with_triples_llm_simple(question, support, client)
    
        if any(kw in response.lower() for kw in bad_keywords):
            print("error")
            help_article = ''
            question_embedding = sentence_model.encode(question)  # Convert the question into an embedding
            similarities = sentence_model.similarity(question_embedding, X).tolist()  # Compute similarity between the query and article embeddings
            top3_idx = np.argsort(similarities[0])[-3:][::-1]  # Select the top 3 articles and get their IDs

            for top_item in top3_idx:
                help_article = help_article + corpus[top_item].get("text", "")
            response,fact = answer_with_article_triple(help_article, question, client)
            if not any(kw in response.lower() for kw in bad_keywords):
                print("corrected")
                update_triples = from_small_text_to_triples(help_article, client)
                added = kb.add_relations_from_triples(update_triples, provenance="hotpotqa_article_42")
                print(fact)
                kb.save(kg_path_update)

                
        facts.append(fact)
        answers.append(response)
        
        print("response",response)

    final_question = item.get("question", "")

    final_response = answer_final_withlogic(final_question, facts, client)

    item["response"] = final_response

    print("final_response",final_response)

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

 
        
        
        
