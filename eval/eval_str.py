from query_eval import *
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate short responses with EM and F1.")
    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to the input JSONL file, e.g., result/wiki_zh_same_length.jsonl")
    return parser.parse_args()

def main():
    args = parse_args()
    file_path = args.file_path

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)

    item_score_all = 0
    f1_score_all = 0
    for item in data:
        missing = [k for k in ('short_response', 'gold_answer') if k not in item or item[k] is None]
        if missing:
            continue
        item_score = exact_match(item['short_response'], item["gold_answer"])
        f1_score = compute_f1(item['short_response'], item["gold_answer"])
        f1_score_all = f1_score_all + f1_score
        item_score_all = item_score_all + item_score
        print(item["gold_answer"], item['short_response'], item_score)

    item_score_all = item_score_all / len(data)
    f1_score_all = f1_score_all / len(data)
    print("item_score_all", item_score_all)
    print("f1_score_all", f1_score_all)

if __name__ == "__main__":
    main()
