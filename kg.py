import pickle
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
from sentence_transformers import SentenceTransformer, util
import re
from tqdm import tqdm
import openai
from collections import defaultdict
from openai import OpenAI
import unicodedata
import numpy as np
import time
import pickle
import os
import time
from typing import List, Dict, Any
DEVICE = "cuda:1"


client = OpenAI(
    base_url="",
    api_key=""
)

def format_facts(facts, max_k=8):
    # Limit length to avoid feeding too much context
    facts = facts[:max_k]
    lines = [f"({h}, {r}, {t})" for (h, r, t) in facts]
    return "\n".join(lines) if lines else "(no facts)"

def canonical(x):
    """Pick the most reliable name from candidates as the canonical entity name."""
    if isinstance(x, str):
        return x
    # x may be like [('score','name'), ...]
    if isinstance(x, list) and x and isinstance(x[0], tuple):
        # Already sorted by similarity descending?
        x_sorted = sorted(x, key=lambda t: t[0], reverse=True)
        return x_sorted[0][1]
    return str(x)

PROMPT_TEMPLATE = """Below are the facts that might be relevant to answer the question:
{facts_block}

Question: {question}
Answer (wrap the final answer in <answer>...</answer> tags only; no other text):

"""


def build_ka_prompt(ans_or_list, question, max_facts=8):
    facts = to_fact_tuples(ans_or_list)
    facts_block = format_facts(facts, max_k=max_facts)
    return PROMPT_TEMPLATE.format(facts_block=facts_block, question=question)



def to_fact_tuples(ans_list):
    """
    Support passing a single ans or a list of ans, output a sequence of fact tuples [(h,r,t), ...].
    Also convert aliases (head_candidates) into 'alias_of' optional facts to enrich context.
    """
    if isinstance(ans_list, dict):
        ans_list = [ans_list]

    facts = []
    for a in ans_list:
        h = canonical(a.get("head"))
        r = canonical(a.get("relation"))
        tails = a.get("tails") or [a.get("tail")]
        tails = [canonical(t) for t in tails if t]
        for t in tails:
            facts.append((h, r, t))

        # Optional: also inject aliases, like "has_part / instance_of" to enrich fact pool
        # Here we simply add head aliases as alias_of for easier model alignment
        hcands = a.get("head_candidates") or []
        main = h
        for _, alias in sorted(hcands, key=lambda x: x[0], reverse=True)[1:3]:
            facts.append((alias, "alias_of", main))
    return facts

def answer_with_article(help_article, question, client):
    prompt = f"Based on the following article, answer the question:\n\nArticle: {help_article}\n\nQuestion: {question}\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def answer_with_article_triple(help_article, question, client):
    prompt = f"""
You are given an article and a question. 
First, extract relevant facts as subject-predicate-object triples (facts). 
Then, based on those facts, provide the final answer.

Format your response as:
Answer: <your answer here>
Facts: <list of triples used>

Article: {help_article}

Question: {question}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
    )
    output = response.choices[0].message.content.strip()

    # Simple parsing
    answer, facts = None, None
    if "Facts:" in output:
        parts = output.split("Facts:")
        answer = parts[0].replace("Answer:", "").strip()
        facts = parts[1].strip()
    else:
        answer = output

    return answer, facts




def answer_final(final_question, questions, answers, client):
    prompt = f"Based on the following facts, answer the final question:\n\n"
    
    for q, a in zip(questions, answers):
        prompt += f"Question: {q}\nAnswer: {a}\n\n"
    prompt += f"Final Question: {final_question}\n\nFinal Answer:"
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def answer_final_withlogic(final_question,facts, client):
    prompt = f"Based on the following facts, answer the final question:\n\n"
    for fact in facts:
        prompt += f"Fact: {fact}\n"
    prompt += f"Final Question: {final_question}\n\nFinal Answer:"
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def build_answer_messages_all_relations(item, original_question, *, topk_relations=None, rationale_max_chars=140):
    """
    Evaluate all relation candidates simultaneously; 
    Output must be a JSON array, corresponding 1-to-1 with relations in the same order.
    Additional constraints are enforced to facilitate post-processing and prevent model escape.
    """
    head = item.get("head", "") or ""
    rel_main = item.get("relation", "") or ""
    # Normalize relation candidates
    rel_cands = [
        (rc[0], rc[1]) if isinstance(rc, (list, tuple)) and len(rc) >= 2
        else (None, rc)  # allow no-score cases
        for rc in (item.get("rel_candidates") or [])
    ]

    # Deduplicate and preserve order, also save scores
    seen = set()
    relations_with_score = []
    # Insert main relation first
    if rel_main and rel_main not in seen:
        relations_with_score.append((None, rel_main))
        seen.add(rel_main)
    # Then add candidates
    for score, rel in rel_cands:
        if rel and rel not in seen:
            relations_with_score.append((score, rel))
            seen.add(rel)

    # Optional: only keep top k (unless you explicitly need all)
    if topk_relations is not None and topk_relations > 0:
        # Keep the first (main relation), truncate the rest
        head_keep = relations_with_score[:1]
        tail_keep = relations_with_score[1:1 + max(0, topk_relations - 1)]
        relations_with_score = head_keep + tail_keep

    relations = [rel for _, rel in relations_with_score]

    # Normalize tails
    tails = item.get("tails")
    if not tails:
        t = item.get("tail")
        tails = [t] if t else []
    tails = [t for t in tails if t]  # filter empty

    head_cands = item.get("head_candidates", []) or []
    rel_cands_scored = item.get("rel_candidates", []) or []

    # Relation+score string block for model
    rel_lines = []
    for score, rel in relations_with_score:
        score_str = "null" if score is None else f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
        rel_lines.append(f"- relation: {rel} | score: {score_str}")
    rel_list_block = "\n".join(rel_lines) if rel_lines else "(none)"

    # Define schema & rules, put strict control in system
    system_block = (
        "You are an information extractor and answerer.\n"
        "Return ONLY valid JSON with NO extra text.\n"
        "Task: For EVERY provided relation, evaluate alignment with the question and head.\n"
        "Output MUST be a JSON ARRAY whose length equals the number of provided relations and preserves their order.\n"
        "Each element MUST be a JSON object with EXACT keys:\n"
        "  answer, confidence, alignment, chosen_head, chosen_relation, chosen_tail(s), rationale, relation_score\n"
        "Constraints:\n"
        "- confidence ∈ {\"high\",\"medium\",\"low\"}\n"
        "- alignment ∈ {\"aligned\",\"corrected\",\"misaligned\"}\n"
        "- chosen_relation MUST equal the evaluated relation string verbatim.\n"
        "- chosen_head MUST be either the original head or one from head_candidates if minimal correction is needed.\n"
        "- If aligned/corrected: answer MUST be selected from `tails` and chosen_tail(s) MUST be a subset of `tails`.\n"
        "- If misaligned: answer MUST be null and chosen_tail(s) MUST be null.\n"
        "- relation_score: echo the numeric score for this relation if provided, otherwise null.\n"
        f"- rationale MUST be ≤ {rationale_max_chars} characters, plain text (no markdown), ≤ 1 sentence.\n"
        "- Do NOT fabricate facts or use external sources.\n"
        "- Use null instead of empty strings.\n"
        "- No additional fields; no markdown; no commentary outside the JSON."
    )

    user_block = (
        f"Original question:\n{original_question}\n\n"
        f"Extracted triple:\nhead: {head}\nrelation: {rel_main}\ntails: {tails}\n\n"
        f"Head candidates (scored desc):\n{head_cands}\n\n"
        f"Relation candidates (scored desc):\n{rel_cands_scored}\n\n"
        "Evaluate ALL of these relations separately, preserving order:\n"
        f"{rel_list_block}\n\n"
        "Evaluation rules:\n"
        "- For EACH relation, independently check if the question asks for THAT relation about the head entity.\n"
        "- If the head slightly mismatches (e.g., ambiguous name), use ONLY head_candidates for minimal correction; mark alignment=\"corrected\".\n"
        "- If you cannot confidently align: alignment=\"misaligned\", answer=null, chosen_tail(s)=null, confidence=\"low\".\n"
        "- If multiple tails are plausible, choose the most specific or commonly accepted one; you may also put multiple in chosen_tail(s) (subset of tails).\n"
        "- Return a JSON ARRAY. No text outside JSON."
    )

    messages = [
        {"role": "system", "content": system_block},
        {"role": "user", "content": user_block},
    ]
    return messages




def build_answer_messages(item, original_question):
    """
    item: 你的实体字典，例如：
      {
        'head': 'Lothair II',
        'relation': 'mother',
        'tail': 'Ermengarde of Tours',
        'head_candidates': [(0.7356, 'Lothair II'), (0.6799, 'Lothair II of Lotharingia'), (0.6515, 'Lothair I')],
        'rel_candidates': [(0.3586, 'mother'), (0.3149, 'father'), (0.1561, 'spouse')],
        'tails': ['Ermengarde of Tours']
      }
    original_question: 原始问题字符串
    """
    head = item.get("head", "")
    relation = item.get("relation", "")
    tails = item.get("tails") or ([item.get("tail")] if item.get("tail") else [])
    head_cands = item.get("head_candidates", [])
    rel_cands = item.get("rel_candidates", [])

    user_block = (
        f"Original question:\n{original_question}\n\n"
        f"Extracted triple:\nhead: {head}\nrelation: {relation}\ntails: {tails}\n\n"
        f"Head candidates (scored desc):\n{head_cands}\n\n"
        f"Relation candidates (scored desc):\n{rel_cands}\n\n"
        "Instructions:\n"
        "- First, check if the original question asks for the same relation (e.g., \"mother of X\") and target entity (X) as in the triple.\n"
        "- If aligned, pick the most appropriate tail from tails as the final answer (if multiple, choose the most specific or commonly accepted).\n"
        "- If partially misaligned (e.g., head mismatch like \"Lothair II of Lotharingia\" vs \"Lothair II\"), minimally correct using the candidates; mark alignment=\"corrected\".\n"
        "- If you cannot align confidently with given candidates, mark alignment=\"misaligned\", answer=null, confidence=\"low\".\n"
        "- Do NOT invent facts or query external sources.\n"
        "- Keep rationale ≤ 1 sentence."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extractor and answerer. "
                "Given a natural-language question and a structured triple candidate (head, relation, tails) plus candidate lists, "
                "1) verify alignment between the question and the triple; "
                "2) if aligned, answer directly using tails; "
                "3) if misaligned, attempt a minimal correction using the provided candidates only; "
                "4) return a short justification (<=1 sentence). "
                "Output MUST be valid JSON with keys: answer, confidence ∈ {high, medium, low}, alignment ∈ {aligned, corrected, misaligned}, chosen_head, chosen_relation, chosen_tail(s), rationale. "
                "Do not include any text outside the JSON."
            )
        },
        # 如需 few-shot，可在这里插入 user/assistant 的演示对：
        # {"role":"user","content": <fewshot_user_text>},
        # {"role":"assistant","content": <fewshot_assistant_json>},
        {
            "role": "user",
            "content": user_block
        }
    ]
    return messages



import re



def build_rewrite_messages_balanced(original_q, answers_prefix, allow_trim=True):
    """
    比 soft 再灵活一点：在确保语义不变的前提下，允许局部重写句子以自然通顺。
    可选：当占位未赋值时，允许“合理删减”多余的虚词/标点。
    """
    pairs = []
    for i, ans in enumerate(answers_prefix, start=1):
        if ans and ans != "Unknown":
            pairs.append(f"{i}: {ans}")
    mapping_block = "\n".join(pairs) if pairs else "None"

    trim_rule = (
        "  • If a k has no value, keep '#k' as-is; "
        "however, you MAY remove immediately adjacent filler words or punctuation "
        "that would otherwise make the sentence broken (e.g., extra commas, empty parentheses).\n"
        if allow_trim else
        "  • If a k has no value, keep '#k' as-is (no trimming).\n"
    )

    system = (
        "You are a precise replacer with gentle rewriting ability.\n"
        "- First, replace tokens '#k' with provided values.\n"
        f"- When a token has no value:\n{trim_rule}"
        "- After replacements, you MAY rephrase locally to ensure fluency and correctness, "
        "but DO NOT change intent, facts, or introduce new content.\n"
        "- Keep structure and formatting largely intact; consolidate only obviously redundant phrases.\n"
        "- Avoid stylistic embellishment; aim for clear, natural language.\n"
        "- Output ONLY inside <rewrite>...</rewrite> with no code fences."
    )

    user = (
        f"Original Question:\n{original_q}\n\n"
        f"Known Replacements (k: value):\n{mapping_block}\n\n"
        "Apply replacements, then gently rewrite locally to make the text natural, "
        "without changing meaning or adding information. "
        "Return ONLY inside <rewrite>...</rewrite>."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
#轻改写
def build_rewrite_messages_soft(original_q, answers_prefix):
    """
    以替换为主，允许“轻度修词”：修正替换导致的语法/标点/空格问题、消除重复词，
    但不改变原意与信息顺序。
    """
    # 生成 k: value 清单（忽略 Unknown/None/""）
    pairs = []
    for i, ans in enumerate(answers_prefix, start=1):
        if ans and ans != "Unknown":
            pairs.append(f"{i}: {ans}")
    mapping_block = "\n".join(pairs) if pairs else "None"

    system = (
        "You are a careful replacer and light rewriter.\n"
        "- Your main job: replace every '#k' using provided values.\n"
        "- If a k has no value, keep '#k' as-is.\n"
        "- After replacements, you MAY do minimal edits:\n"
        "  • fix spacing/punctuation/typos introduced by replacement;\n"
        "  • remove duplicated words;\n"
        "  • adjust number/gender/plural forms if the language requires it;\n"
        "  • delete dangling punctuation or brackets left empty.\n"
        "- Do NOT change meaning, claims, ordering of facts, or add new info.\n"
        "- Preserve formatting unless readability clearly improves.\n"
        "- Output ONLY inside <rewrite>...</rewrite> with no code fences."
    )

    user = (
        f"Original Question:\n{original_q}\n\n"
        f"Known Replacements (k: value):\n{mapping_block}\n\n"
        "Perform replacements, then apply only the minimal edits listed. "
        "Return ONLY inside <rewrite>...</rewrite>."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def build_rewrite_messages_replace_only(original_q, answers_prefix):
    """
    仅做 '#k' -> 已知答案 的字面替换；严禁改写其它内容。
    """
    # 把已知答案做成 k: value 的映射清单；Unknown/None/"" 不替换
    pairs = []
    for i, ans in enumerate(answers_prefix, start=1):
        if ans and ans != "Unknown":
            pairs.append(f"{i}: {ans}")
    mapping_block = "\n".join(pairs) if pairs else "None"

    system = (
        "You are a STRICT replacer.\n"
        "- Copy the original question EXACTLY.\n"
        "- Replace every occurrence of '#k' with the provided value for k.\n"
        "- If a k has no value provided, leave '#k' UNCHANGED.\n"
        "- Do NOT paraphrase, reorder, add, or remove any other text.\n"
        "- Output ONLY inside <rewrite>...</rewrite> with no code fences."
    )

    user = (
        f"Original Question:\n{original_q}\n\n"
        f"Known Replacements (k: value):\n{mapping_block}\n\n"
        "Apply ONLY the replacements. Return ONLY inside <rewrite>...</rewrite>."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_rewrite_tag(text: str) -> str:
    """从模型输出中提取 <rewrite>...</rewrite>；若无则返回原文去除围栏。"""
    if not text:
        return ""
    # 去掉可能的代码围栏
    cleaned = text.replace("```", "").replace("```json", "").strip()
    m = re.search(r"<rewrite>\s*(.*?)\s*</rewrite>", cleaned, flags=re.S | re.I)
    return m.group(1).strip() if m else cleaned.strip()


def rewrite_with_llm(original_q, answers_prefix, client, model="gpt-4o-mini-2024-07-18"):
    messages = build_rewrite_messages_replace_only(original_q, answers_prefix)
    resp = client.chat.completions.create(model=model, messages=messages,temperature=0)
    rewrited = extract_rewrite_tag(resp.choices[0].message.content)

    # 保险：如果模型没按要求输出，做一次本地兜底的“只替换不改写”
    if not rewrited or "<rewrite>" in rewrited.lower():
        out = original_q
        for i, ans in enumerate(answers_prefix, start=1):
            if ans and ans != "Unknown":
                out = re.sub(fr'(?<!\w)#{i}(?!\w)', str(ans), out)
        rewrited = out
    return rewrited









def build_rewrite_messages(original_q, subqs, answers_prefix):
    """
    original_q: 当前要处理的子问题（原始分解问题文本）
    subqs: 全部子问题列表
    answers_prefix: 之前已得到的答案列表（长度 == 当前 idx）
    """
    # 组装已知(Q,A)对，只到当前idx之前
    lines = []
    for i in range(len(answers_prefix)):
        q_i = subqs[i]
        a_i = answers_prefix[i] if answers_prefix[i] not in [None, ""] else "Unknown"
        lines.append(f"#{i+1} Q: {q_i}\n#{i+1} A: {a_i}")
    qa_block = "\n".join(lines) if lines else "None"

    # few-shot（可按需增删）
    fewshot_user = (
        "Original Question:\n"
        "What is the name of the famous bridge in the birth city of the composer of Scanderbeg?\n\n"
        "Known Sub-questions & Answers:\n"
        "#1 Q: Who is the composer of Scanderbeg?\n"
        "#1 A: Antonio Vivaldi\n"
        "#2 Q: What is the birth city of #1?\n"
        "#2 A: Venice\n\n"
        "Instruction:\n"
        "Rewrite a single coherent question that incorporates any known answers (#1, #2, ...) to make it self-contained. "
        "Use ONLY the provided answers; do not fabricate missing facts. "
        "Return ONLY inside <rewrite>...</rewrite>."
    )
    fewshot_assistant = "<rewrite>What is the name of the famous bridge in Venice, the birth city of Antonio Vivaldi?</rewrite>"

    current_user = (
        f"Original Question:\n{original_q}\n\n"
        "Known Sub-questions & Answers:\n"
        f"{qa_block}\n\n"
        "Instruction:\n"
        "Rewrite a single coherent question that incorporates any known answers (#1, #2, ...) to make it self-contained. "
        "If some answers are Unknown, keep that part generic. "
        "Return ONLY inside <rewrite>...</rewrite>."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise question rewriter. Produce ONE self-contained interrogative sentence that integrates "
                "any known answers (#1, #2, ...). Do NOT invent facts. Output ONLY inside <rewrite>...</rewrite>."
            ),
        },
        {"role": "user", "content": fewshot_user},
        {"role": "assistant", "content": fewshot_assistant},
        {"role": "user", "content": current_user},
    ]
    return messages


def link_entity(entity, nlp):
    pattern = r'Q\d+'
    name = entity
    linking = re.search(pattern, str(nlp(entity)._.linkedEntities))
    if linking:
        linking = linking.group(0) # Q?
    else:
        linking = name
    return linking

def get_scores(question, objects, model, threshold=None, topk=None):
    """
    Return [(score, candidate), ...]; optional threshold and top-k.
    """
    if objects is None:
        return []
    if isinstance(objects, str):
        candidates = [objects]
    elif isinstance(objects, set):
        candidates = sorted(objects)           # set -> stable order
    else:
        candidates = list(objects)

    # Clean empty values
    candidates = [str(x).strip() for x in candidates if x is not None and str(x).strip() != ""]
    if not candidates:
        return []

    # Encode + normalize (dot product == cosine similarity)
    q_emb   = model.encode(question,  convert_to_tensor=True, normalize_embeddings=True)
    obj_emb = model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(q_emb, obj_emb).squeeze(0)   # (N,)
    scores_all = sims.cpu().tolist()

    pairs = [(score, cand) for score, cand in zip(scores_all, candidates)]
    if threshold is not None:
        pairs = [p for p in pairs if p[0] >= threshold]
    pairs.sort(key=lambda x: x[0], reverse=True)
    if topk is not None and topk > 0:
        pairs = pairs[:topk]
    return pairs





def build_indices(relations):
    """
    relations: List[{'head':..., 'type':..., 'tail':...}]
    Returns:
      nodes: all node names (set)
      rel_index: dict[str -> List[str]]     # candidate relation list for each head
      triple_index: dict[(head, rel) -> List[str]]  # (head, rel) -> tails
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
    # Normalize sets to lists
    rel_index = {k: sorted(list(v)) for k, v in rel_index.items()}
    return nodes, rel_index, triple_index

def triples_to_strings(relations):
    fact_strs = []
    for r in relations:
        if isinstance(r, dict):
            # Adapt dict
            s, p, o = r.get("head"), r.get("type") or r.get("relation"), r.get("tail")
        elif isinstance(r, tuple) and len(r) == 3:
            # Adapt (s, p, o) tuple
            s, p, o = r
        elif hasattr(r, "head") and hasattr(r, "relation") and hasattr(r, "tail"):
            # Adapt Relation class
            s, p, o = r.head, r.relation, r.tail
        else:
            # Unknown format, skip
            continue

        fact_strs.append(f"({s}, {p}, {o})")
    return fact_strs

import re

def answer_with_triples_llm_final(
    question,
    support_facts,
    client,
    subqas=None,
    model="gpt-4o-mini-2024-07-18",
):
    """
    Only allow answering based on support_facts and subqas;
    if information is insufficient, must output 'can not solve'.
    Output uses XML tags: <answer>…</answer> and optional <evidence>…</evidence>.
    """
    # 1) Build the facts block (numbered for easy reference)
    facts_list = support_facts or []
    facts_block = (
        "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts_list))
        if facts_list else "(none)"
    )

    # 2) Sub-QA block (if provided)
    subqas = subqas or []
    subqa_block = (
        "\n".join(
            f"{i+1}. Q: {sq.get('q', sq[0] if isinstance(sq, (list, tuple)) else '')}\n"
            f"   A: {sq.get('a', sq[1] if isinstance(sq, (list, tuple)) else '')}"
            for i, sq in enumerate(subqas)
        )
        if subqas else "(none)"
    )

    # 3) Prompt (require internal reasoning; do not reveal chain-of-thought; if insufficient then can not solve)
    user_prompt = f"""
            You are a precise QA system. Use ONLY the information below.

            Facts:
            {facts_block}

            Sub-questions and their answers:
            {subqa_block}

            Current question:
            {question}

            Instructions:
            - Do any necessary reasoning internally; do NOT reveal your chain-of-thought.
            - Answer ONLY if the facts and sub-answers are sufficient.
            - If insufficient, output exactly: can not solve
            - Output format (XML) strictly as below:
            <answer>final answer in one sentence or phrase</answer>
            <evidence>comma-separated fact indices you used (e.g., 2,5)</evidence>
            """.strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful QA assistant. "
                "Use only provided facts and sub-answers. "
                "If insufficient, reply exactly: can not solve. "
                "Never reveal chain-of-thought."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()

    # 4) Parse <answer> and <evidence>
    m_ans = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    m_evd = re.search(r"<evidence>\s*(.*?)\s*</evidence>", text, flags=re.S | re.I)

    if m_ans:
        answer = m_ans.group(1).strip()
    else:
        # Fallback: if model did not return XML, treat the whole text as answer
        answer = text

    evidence = []
    if m_evd:
        raw = m_evd.group(1)
        # Parse comma-separated indices
        evidence = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]

    # Unified handling for “cannot answer”
    if answer.strip().lower() in ["can not solve", "cant solve", "cannot solve"]:
        return answer
    return answer
  


def answer_with_triples_llm(question, support_facts, client, model="gpt-4o-mini-2024-07-18"):
    """
    Only allow the model to answer based on support_facts; if facts are insufficient, must answer "can not solve".
    """
    facts_block = "\n".join(f"- {f}" for f in support_facts) if support_facts else "(none)"

    prompt = f"""
        You are a precise QA system. 
        You can answer the question using the facts below. 

        Facts:
        {facts_block}

        Question: {question}
        Instructions:
        - If the facts are enough, answer concisely.
        - If the facts do NOT contain enough information to answer, output exactly: can not solve
        Answer (concise):
        """.strip()


    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def answer_with_triples_llm_simple(question, support_facts, client, model="gpt-4o-mini-2024-07-18"):
    """
    Only allow the model to answer based on support_facts; if facts are insufficient, must answer "can not solve".
    """
    facts_block = "\n".join(f"- {f}" for f in support_facts) if support_facts else "(none)"

    # prompt = f"""
    #     You are a precise QA system. 
    #     You can answer the question using the facts below**. 

    #     Facts:
    #     {facts_block}

    #     Question: {question}
    #     Answer (concise):
    #     """.strip()
    prompt = f"""
    You are a careful reasoning assistant. 
    Use the following facts to answer the question. 

    Facts:
    {facts_block}

    Question: {question}

    Instructions:
    - Give a brief answer.
    - Then list only the facts you actually used to reach the answer.
    - Format strictly like this:

    Answer: <your short answer>
    Used facts:
    """

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    m = re.search(r"Answer:\s*(.+)", resp.choices[0].message.content.strip()).group(1).strip()
    used_facts = re.findall(r"-\s*(.+)", resp.choices[0].message.content.strip())
    if not used_facts:
        used_facts = support_facts[:3]
    return m,used_facts


def select_support_triples(question, relations, sentence_model, topk=10):
    """Use sentence embeddings to select the top-k triples most relevant to the question."""
    fact_strs = triples_to_strings(relations)
    if not fact_strs:
        return []

    q_emb = sentence_model.encode(question)
    X_fact = sentence_model.encode(fact_strs)          # [n, d]
    # Similarity: cosine (available in most sentence embedding libs)
    sims = sentence_model.similarity(q_emb, X_fact).ravel().tolist()
    idx = np.argsort(sims)[-topk:][::-1]
    return [fact_strs[i] for i in idx]



def _norm(s: str) -> str:
    # Normalize: NFKC + lowercase + strip
    return unicodedata.normalize("NFKC", (s or "")).strip().lower()

def _ensure_sorted(scores):
    """
    Compatible with two shapes:
    - [(score, item), ...]
    - [(item, score), ...]
    Normalize to [(score, item), ...] and sort descending.
    """
    if not scores:
        return []
    a, b = scores[0]
    # Guess which element is the score
    if isinstance(a, (int, float)) and not isinstance(b, (int, float)):
        pairs = scores
    elif isinstance(b, (int, float)) and not isinstance(a, (int, float)):
        pairs = [(b, a) for (a, b) in scores]
    else:
        # If uncertain, try casting both to float; if fails, return original list
        try:
            pairs = [(float(a), b) for (a, b) in scores]
        except Exception:
            pairs = scores
    return sorted(pairs, key=lambda x: x[0], reverse=True)



def retrieve_from_graph_no_link_v2(
    question: str,
    embed_model,
    nodes, rel_index, triple_index,
    entity_threshold: float = 0.2,
    rel_threshold: float = 0.1,
    topk_heads: int = 5,
    alpha: float = 0.3,  # Fusion weight: pair_score * (1 - alpha) + head_score * alpha
):
    """
    Select candidates based on similarity between (head, relation) pairs and the question, then fetch tails.
    - First select topk_heads heads (coarse filter)
    - For each head’s outgoing relations, build (head, rel) pair text and compute similarity to the question
    - Choose the highest-scoring (head, relation) pair
    - Then fetch tails from triple_index and re-rank tails using (question + head + rel)
    """

    if not question or not nodes:
        return None

    q_norm = _norm(question)  # Unified normalization

    # Normalize nodes and create mapping
    norm2raw_node, norm_nodes = {}, []
    for n in nodes:
        nn = _norm(n)
        if nn:
            norm2raw_node[nn] = n
            norm_nodes.append(nn)

    # Normalize relation index
    norm_rel_index = {}
    for h, rels in rel_index.items():
        hn = _norm(h)
        if not hn:
            continue
        seen, clean = set(), []
        for r in rels:
            rn = _norm(r)
            if rn and rn not in seen:
                seen.add(rn)
                clean.append(rn)
        if clean:
            norm_rel_index[hn] = clean

    # Normalize triple index
    norm_triple_index = {}
    for (h, r), tails in triple_index.items():
        hn, rn = _norm(h), _norm(r)
        if not hn or not rn:
            continue
        seen, clean_tails = set(), []
        for t in tails:
            tn = _norm(t)
            if tn and tn not in seen:
                seen.add(tn)
                clean_tails.append(tn)
        if clean_tails:
            norm_triple_index[(hn, rn)] = clean_tails

    # 1) Coarse head selection (by similarity to the question)
    ent_scores_raw = get_scores(q_norm, norm_nodes, embed_model, threshold=None, topk=None)
    ent_scores = _ensure_sorted(ent_scores_raw)

    # Threshold filter + fallback to at least one
    ent_scores = [p for p in ent_scores if p[0] >= entity_threshold] or ent_scores[:1]
    if not ent_scores:
        return None

    # Limit number of heads to combine to control cost
    head_pool = [h for _, h in ent_scores[:max(1, topk_heads)]]

    # 2) Build (head, rel) pairs and compute pair similarity
    pair_texts = []        # texts for scoring
    pair_keys = []         # aligned (head_norm, rel_norm) for each text
    pair_headpriors = []   # head prior score (from ent_scores)
    head2score = {h: s for s, h in ent_scores}

    for hn in head_pool:
        rels = norm_rel_index.get(hn) or []
        for rn in rels:
            # Pair text template; you can change this format
            pair_text = f"{hn} [rel:{rn}]"
            pair_texts.append(pair_text)
            pair_keys.append((hn, rn))
            pair_headpriors.append(head2score.get(hn, 0.0))

    if not pair_texts:
        return None

    pair_scores_raw = get_scores(q_norm, pair_texts, embed_model, threshold=None, topk=None)
    pair_scores = _ensure_sorted(pair_scores_raw)

    # 3) Fused score (optional): pair score + head prior
    #    This helps better heads win when pair scores are close
    fused = []
    score_map = {txt: s for s, txt in pair_scores}
    for txt, (hn, rn), hprior in zip(pair_texts, pair_keys, pair_headpriors):
        ps = score_map.get(txt, 0.0)
        fused_score = (1 - alpha) * ps + alpha * hprior
        fused.append((fused_score, hn, rn, ps, hprior))

    fused.sort(key=lambda x: x[0], reverse=True)

    # Optional: filter by relation threshold (on fused or raw pair score)
    fused = [it for it in fused if it[0] >= rel_threshold] or (fused[:1] if fused else [])
    if not fused:
        return None

    best_score, head_norm, rel_norm, pair_score, head_prior = fused[0]
    head_raw = norm2raw_node.get(head_norm, head_norm)
    rel_raw = rel_norm  # If you need original text, keep a map when building indices

    # 4) Fetch tails
    tails = norm_triple_index.get((head_norm, rel_norm), [])
    if not tails:
        return None

    # 5) Re-rank tails
    rerank_query = f"{q_norm} [head:{head_norm}] [rel:{rel_norm}]"
    tail_scores_raw = get_scores(rerank_query, tails, embed_model, threshold=None, topk=None)
    tail_scores = _ensure_sorted(tail_scores_raw)
    best_tail_norm = (tail_scores[0][1] if tail_scores else tails[0])
    best_tail_raw = best_tail_norm

    # 6) Additional diagnostics: keep top-3 heads and top-3 pairs (for debugging/display)
    head_candidates = ent_scores[:3]
    # Show top pairs as (score, "rel") — or keep a structured dict if preferred
    rel_candidates = []
    for sc, hn, rn, ps, hp in fused[:3]:
        rel_candidates.append((float(sc), rn))  # Alternatively: {"fused": sc, "pair": ps, "head_prior": hp, "rel": rn, "head": hn}

    return {
        "head": head_raw,
        "relation": rel_raw,
        "tail": best_tail_raw,
        "head_candidates": head_candidates,
        "rel_candidates": rel_candidates,
        "tails": tails,
        "diagnostics": {
            "best_pair_fused": float(best_score),
            "best_pair_raw": float(pair_score),
            "best_head_prior": float(head_prior),
        }
    }





def retrieve_from_graph_no_link(
    question: str,
    embed_model,
    nodes, rel_index, triple_index,
    entity_threshold: float = 0.2,
    rel_threshold: float = 0.1
):
    if not question or not nodes:
        return None

    q_norm = _norm(question)

    # Normalize: also create mappings for nodes/keys in indices
    norm2raw_node = {}
    norm_nodes = []
    for n in nodes:
        nn = _norm(n)
        if not nn:
            continue
        norm2raw_node[nn] = n
        norm_nodes.append(nn)

    # Normalize relation index and triple index
    norm_rel_index = {}
    for h, rels in rel_index.items():
        hn = _norm(h)
        if not hn:
            continue
        # Filter empties, deduplicate + normalize
        clean = []
        seen = set()
        for r in rels:
            rn = _norm(r)
            if rn and rn not in seen:
                seen.add(rn)
                clean.append(rn)
        norm_rel_index[hn] = clean

    norm_triple_index = {}
    for (h, r), tails in triple_index.items():
        hn, rn = _norm(h), _norm(r)
        if not hn or not rn:
            continue
        clean_tails = []
        seen = set()
        for t in tails:
            tn = _norm(t)
            if tn and tn not in seen:
                seen.add(tn)
                clean_tails.append(tn)
        norm_triple_index[(hn, rn)] = clean_tails

    # 1) Select head entity
    ent_scores_raw = get_scores(q_norm, norm_nodes, embed_model, threshold=None, topk=None)
    ent_scores = _ensure_sorted(ent_scores_raw)

    # Threshold filter; fallback to at least one
    ent_scores = [p for p in ent_scores if p[0] >= entity_threshold] or ent_scores[:1]
    if not ent_scores:
        return None
    head_norm = ent_scores[0][1]
    head_raw = norm2raw_node.get(head_norm, head_norm)

    # 2) Select relation (from head’s outgoing edges)
    cand_rels = norm_rel_index.get(head_norm, [])
    if not cand_rels:
        return None

    rel_scores_raw = get_scores(q_norm, cand_rels, embed_model, threshold=None, topk=None)
    rel_scores = _ensure_sorted(rel_scores_raw)
    rel_scores = [p for p in rel_scores if p[0] >= rel_threshold] or rel_scores[:1]
    if not rel_scores:
        return None
    rel_norm = rel_scores[0][1]
    rel_raw = rel_norm  # If needed, keep original text mapping when building indices

    # 3) Fetch tails
    tails = norm_triple_index.get((head_norm, rel_norm), [])
    if not tails:
        return None

    # 4) Re-rank tails (combine head and question to improve disambiguation)
    rerank_query = f"{q_norm} [head:{head_norm}] [rel:{rel_norm}]"
    tail_scores_raw = get_scores(rerank_query, tails, embed_model, threshold=None, topk=None)
    tail_scores = _ensure_sorted(tail_scores_raw)

    best_tail_norm = (tail_scores[0][1] if tail_scores else tails[0])
    best_tail_raw = best_tail_norm  # Likewise, keep original text if needed

    # Return more diagnostics
    return {
        "head": head_raw,
        "relation": rel_raw,
        "tail": best_tail_raw,
        "head_candidates": ent_scores[:3],
        "rel_candidates": rel_scores[:3],
        "tails": tails
    }



def extract_relations_from_model_output(text):
    """
    Directly parse a JSON array returned by the model; each element contains subject/predicate/object.
    """
    try:
        relations = json.loads(text)
        # Ensure it is a list and contains correct fields
        clean_relations = []
        for r in relations:
            if isinstance(r, dict) and "subject" in r and "predicate" in r and "object" in r:
                clean_relations.append({
                    "head": r["subject"].strip(),
                    "type": r["predicate"].strip(),
                    "tail": r["object"].strip()
                })
        return clean_relations
    except json.JSONDecodeError:
        print("JSON parse failed, raw text:", text)
        return []

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
    

def from_small_text_to_kb(text, verbose=False):
    kb = KB()

    # Force the model to output JSON only
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

    # Step 1. Clean common code fence markers
    cleaned = content.replace("```json", "").replace("```", "").strip()

    # Step 2. Use regex to locate the first JSON array
    match = re.search(r"\[.*\]", cleaned, re.S)
    if match:
        cleaned = match.group(0)

    # Step 3. Try parsing JSON
    triples = []
    try:
        triples = json.loads(cleaned)
    except json.JSONDecodeError as e:
        if verbose:
            print("JSON parse failed:", e)
            print("raw:", content)

    # Step 4. Add into KB
    for triple in triples:
        if isinstance(triple, dict) and "subject" in triple and "predicate" in triple and "object" in triple:
            kb.add_relation({
                "head": triple["subject"].strip(),
                "type": triple["predicate"].strip(),
                "tail": triple["object"].strip()
            })
    

    return kb

def from_small_text_to_triples(text, client, verbose=False):
    """
    Input a piece of text, call LLM to extract triples, and output a list where each element is:
    {"head": str, "relation": str, "tail": str, "confidence": float}
    """
    prompt = f"""
    Please extract factual knowledge triples from the following text.
    Return ONLY a valid JSON array (no extra words, no code fences).
    Each element must be in the form:
    {{"head": "", "relation": "", "tail": "", "confidence": 0.x}}
    
    Text: {text}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()

    # Step 1. Clean common code fence markers
    cleaned = content.replace("```json", "").replace("```", "").strip()

    # Step 2. Use regex to locate the first JSON array
    match = re.search(r"\[.*\]", cleaned, re.S)
    if match:
        cleaned = match.group(0)

    # Step 3. Try parsing JSON
    triples = []
    try:
        triples = json.loads(cleaned)
    except json.JSONDecodeError as e:
        if verbose:
            print("JSON parse failed:", e)
            print("raw:", content)

    # Step 4. Normalize output
    new_triples = []
    for t in triples:
        if isinstance(t, dict):
            head = t.get("head") or t.get("subject") or ""
            relation = t.get("relation") or t.get("predicate") or ""
            tail = t.get("tail") or t.get("object") or ""
            conf = float(t.get("confidence", 0.9))  # default 0.9
            if head and relation and tail:
                new_triples.append({
                    "head": head.strip(),
                    "relation": relation.strip(),
                    "tail": tail.strip(),
                    "confidence": conf
                })

    return new_triples



class KnowledgeBase:
    def __init__(self):
        self.relations = []

    def add_relation(self, relation):
        self.relations.append(relation)



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


class DynamicKnowledgeBase:
    def __init__(self, path: str = None):
        self.relations: List[Relation] = []
        self.path = path
        if path and os.path.exists(path):
            self.load(path)

    # ===== Core functions =====
    def add_relation(self, relation: Relation) -> bool:
        """Add a single relation, avoiding duplicates."""
        for r in self.relations:
            if r.head == relation.head and r.relation == relation.relation and r.tail == relation.tail:
                return False
        self.relations.append(relation)
        return True

    def add_relations_from_triples(self, triples: List[Dict[str, Any]],
                                   provenance="article", conf_threshold=0.55) -> int:
        """Insert from a list of triple dicts (commonly from LLM extraction)."""
        count = 0
        for t in triples:
            if t.get("confidence", 1.0) < conf_threshold:
                continue
            rel = Relation(
                head=t["head"],
                relation=t.get("relation") or t.get("type", ""),
                tail=t["tail"],
                confidence=t.get("confidence", 1.0),
                provenance=provenance,
                timestamp=time.time()
            )
            if self.add_relation(rel):
                count += 1
        return count

    def search(self, entity: str) -> List[Relation]:
        """Find all relations related to an entity."""
        return [r for r in self.relations if r.head == entity or r.tail == entity]

    def save(self, path: str = None):
        """
    Persist the current KB as a snapshot in the format:
      (nodes: set[str], rel_index: dict[str, List[str]], triple_index: dict[(str,str), List[str]])
    """

        if path is None:
            path = self.path
        assert path, "save() requires a path or self.path set in __init__"

        # Relation -> dict to fit the input of build_indices
        rel_dicts = [
            {"head": rel.head, "type": rel.relation, "tail": rel.tail}
            for rel in self.relations
        ]

        # Build snapshot
        nodes, rel_index, triple_index = build_indices(rel_dicts)

        # Atomic write: write to .tmp then replace
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump((nodes, rel_index, triple_index), f)

        os.replace(tmp_path, path)
        print(f"[KB] Saved snapshot: |nodes|={len(nodes)}, |heads|={len(rel_index)}, |pairs|={len(triple_index)} -> {path}")



def build_kb_from_pickle(kg_path: str, out_path: str = None) -> DynamicKnowledgeBase:
    with open(kg_path, "rb") as f:
        entities_loaded, attributes_loaded, triples_loaded = pickle.load(f)

    kb = DynamicKnowledgeBase()
    total = len(triples_loaded)
 
    for (subject, predicate), obj in tqdm(triples_loaded.items(), total=total, desc="Expanding triples"):
        if isinstance(obj, str):
            kb.add_relation(Relation(subject.strip(), predicate.strip(), obj.strip()))
        elif isinstance(obj, list):
            for o in obj:
                kb.add_relation(Relation(subject.strip(), predicate.strip(), str(o).strip()))
        else:
            kb.add_relation(Relation(subject.strip(), predicate.strip(), str(obj).strip()))
    return kb
