## Quickstart

```
cd ch01

source .venv/bin/activate

pip install --upgrade pip

## For WSL2 Ubuntu
pip install 'httpx[socks]'

pip install langchain-community langchain-openai langchain faiss-cpu sentence-transformers

python rag_cloud_security_qa.py

```