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

python create_kg/create_kg.py 
  --data_path data/2wikimultihopqa_corpus.json 
  --out_path data/kg/2wikimultihopqa_corpus.pkl 
  --ckpt_path /data/kg/2wikimultihopqa_corpus.pkl

```
