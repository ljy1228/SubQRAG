import json
from openai import OpenAI

client = OpenAI(
    base_url="",
    api_key="",   
)


import os, json
from openai import OpenAI


MODEL = "gpt-4o-mini-2024-07-18"
NUM_CANDIDATES = 5          
TEMPERATURE = 0.8          
TOP_P = 1
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0

in_path = "result/wiki/wiki.jsonl"
out_path = "result/wiki/wiki_same_length.jsonl"


MODEL = "gpt-4o-mini-2024-07-18"



def tokenize_words(s: str):
    if not isinstance(s, str):
        return []
    s = s.replace("\n", " ")
    s = " ".join(s.split())
    return s.split()

def words_to_text(words):
    return " ".join(words)

def enforce_word_count(text: str, n: int) -> str:
    words = tokenize_words(text)
    if n <= 0:
        return ""
    if len(words) > n:
        words = words[:n]
    elif len(words) < n:
        if len(words) == 0:
            words = ["unknown"] * n
        else:
            last = words[-1]
            words = words + [last] * (n - len(words))
    return words_to_text(words)

def unique_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

with open(in_path, "r", encoding="utf-8") as fin, \
     open(out_path, "w", encoding="utf-8") as fout:

    for lineno, line in enumerate(fin, start=1):

        item = json.loads(line)
        question = item.get("question", "")
        long_answer = item.get("response", "")
        gold_answer = item.get("answer", "")

        target_word_count = len(tokenize_words(gold_answer))
        candidates = []

        try:
            if target_word_count <= 0:
                resp = client.chat.completions.create(
                    model=MODEL,
                    n=NUM_CANDIDATES,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    presence_penalty=PRESENCE_PENALTY,
                    frequency_penalty=FREQUENCY_PENALTY,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise assistant. "
                                "Provide the most accurate and shortest possible answer. "
                                "Do not explain or add extra words. Output a single line."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n"
                                f"Background (may be noisy): {long_answer}\n\n"
                                "Give only the most accurate and shortest possible answer."
                            ),
                        },
                    ],
                )
                raw_texts = [c.message.content for c in resp.choices]
                candidates = unique_keep_order([" ".join(tokenize_words(t or "")) for t in raw_texts if t])

            else:
                N = target_word_count
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Follow formatting constraints exactly. "
                            f"Return EXACTLY {N} words, separated by single spaces. "
                            "No explanations, no quotes, no punctuation-only tokens, no prefixes/suffixes."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Answer the question as accurately as possible in English.\n"
                            f"You MUST output EXACTLY {N} words.\n\n"
                            f"Question: {question}\n"
                            f"Background (may be helpful or noisy): {long_answer}\n\n"
                            f"Return exactly {N} words and nothing else."
                        ),
                    },
                ]
                resp = client.chat.completions.create(
                    model=MODEL,
                    n=NUM_CANDIDATES,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    presence_penalty=PRESENCE_PENALTY,
                    frequency_penalty=FREQUENCY_PENALTY,
                    messages=messages,
                )
                raw_texts = [c.message.content for c in resp.choices]
                corrected = [enforce_word_count(t or "", N) for t in raw_texts]
                candidates = unique_keep_order([" ".join(tokenize_words(t)) for t in corrected])

        except Exception as e:
            candidates = ["unknown"]

        preferred = candidates[0] if candidates else "unknown"
        item["short_response"] = preferred
        item["short_response_candidates"] = candidates
        item["gold_answer"] = gold_answer



        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
