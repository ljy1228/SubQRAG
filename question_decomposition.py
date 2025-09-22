import os
import json
import time
from openai import OpenAI

client = OpenAI(
  api_key="",
)


data_question_path = ""  # your actual local data file path.例如2wikimultihopqa.json

with open(data_question_path, "r", encoding="utf-8") as f:
    data = json.load(f)   

out_path = ""  #output_path

with open(out_path, "a", encoding="utf-8") as fout:
    for i, item in enumerate(data, 1):  

        q = item.get("question", "")
        _id = item.get("id", None)
        answer = item.get("answers", "")[0] 

        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "<YOUR_SITE_URL>",
                    "X-Title": "<YOUR_SITE_NAME>",
                },
                extra_body={},
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in question decomposition. "
                            "Output ONLY a numbered list of interrogative sub-questions. "
                            "Rules: (1) Each line is a single question ending with '?'. "
                            "No explanations. No summaries. "
                            "Later questions may reference earlier answers via #1, #2, etc. "
                            "If the original question has contradictory assumptions, "
                            "first rewrite it minimally to be consistent, then decompose."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Example:\n"
                            "Original Question: What is the name of the famous bridge in the birth city of the composer of Scanderbeg?\n"
                            "1. Who is the composer of Scanderbeg?\n"
                            "2. What is the birth city of #1?\n"
                            "3. What is the name of the famous bridge in #2?\n\n"
                            f"Now decompose the following:\nOriginal Question: {q}"
                        )
                    }
                ]
            )

            decomposition = completion.choices[0].message.content.strip()
            record = {
                "_id": _id,
                "question": q,
                "decomposition": decomposition,
                "answer":answer
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}] saved: {_id}")
            time.sleep(0.2)

        except Exception as e:
        
            err_rec = {"_id": _id, "question": q, "error": str(e)}
            fout.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
            print(f"[{i}] error: {_id} -> {e}")
            time.sleep(1.0)

print(f"All done. Results in: {out_path}")