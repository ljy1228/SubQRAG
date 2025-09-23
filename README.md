# SubQRAG
<h2 align="center">SUBQRAG: SUB-QUESTION DRIVEN DYNAMIC GRAPH RAG</h3>

<p align="center">
  <img src="image/image.png" width="55%" style="max-width: 300px;">
</p>


## Installation

```sh
conda create -n subqrag python=3.10
conda activate subqrag
cd SubQRAG
pip install -r requirements.txt
```
## 📚 Datasets
We follow the same dataset as HippoRAG. 
[<img align="center" src="https://img.shields.io/badge/🤗 Dataset-HippoRAG 2-yellow" />](https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/main)

```sh
https://huggingface.co/datasets/osunlp/HippoRAG_2

```
## ✨ Offline Indexing (Pre-constructing a Knowledge Graph)

```sh

python create_kg/create_kg.py --data_path data/2wikimultihopqa_corpus.json --out_path data/kg/2wikimultihopqa_corpus.pkl   --ckpt_path /data/kg/2wikimultihopqa_corpus.pkl 

```
## 🚀 Decompose the subproblems

```sh

python question_decomposition.py --data_path data/2wikimultihopqa.json --out_path decompose_data/decompose_2wikimultihopqa.jsonl

```

## 👉 Retrieval+Update+Generation

Embedding

```sh
python embedding.py --data_path data/2wikimultihopqa_corpus.json --emb_path embdding_all/wiki_embeddings.npy --meta_path wiki_metajsonl
```
Retrieval+Update+Generation

```sh

python subq_rag.py --decomp_path decompose_data/decompose_2wikimultihopqa.jsonl --corpus_path data/2wikimultihopqa_corpus.json --emb_path embdding_all/wiki_embeddings.npy --kg_pickle_in kg_dataset/dynamic_kb.pkl --kg_pickle_out kg_dataset/2wikimultihopqa_dynamic_kb.pkl --sent_model_path sentence-transformers/all-MiniLM-L6-v2 --out_path result/2wiki/2wiki_update_graph.jsonl --openai_base_url "" --openai_api_key "$OPENAI_API_KEY"
```
## 📜 Evaluation

Extract the answer
```sh
python make_short_answers.py --in_path result/wiki/wiki.jsonl --out_path result/wiki/wiki_same_length.jsonl --base_url "" --api_key "$OPENAI_API_KEY" --model gpt-4o-mini
```
EM + F1
```sh
python eval/eval_str.py --file_path result/wiki_zh_same_length.jsonl
```

If you find this project helpful, feel free to ⭐️ Star and 🔁 Fork it!