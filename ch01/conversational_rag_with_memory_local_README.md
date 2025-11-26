## Quickstart for conversational_rag_with_memory_local.py on macOS iter2 zsh

```
**第1步：创建虚拟环境**
```bash
cd /Users/binwu/OOR-local/katas/lab-book-rag-2-in-action/ch01
python3.12 -m venv .venv
```

**第2步：激活虚拟环境**
```bash
source .venv/bin/activate
```

**第3步：升级pip**
```bash
pip install --upgrade pip
```

**第4步：安装所需的依赖包**

For `conversational_rag_with_memory_local.py` (recommended):
```bash
pip install langchain-community langchain-openai faiss-cpu sentence-transformers

## For WSL2 Ubuntu
pip install 'httpx[socks]'
```

Or for `conversational_rag_with_memory.py` (requires OpenAI tiktoken):
```bash
pip install langchain-community langchain-openai faiss-cpu tiktoken
```

**第5步：准备DeepSeek API密钥**

这个脚本使用 DeepSeek-reasoner API。在运行时，脚本会提示你输入 DeepSeek API 密钥。
请确保你已经注册并获取了 DeepSeek API 密钥（访问 https://platform.deepseek.com）。

**第6步：运行脚本**
```bash
python conversational_rag_with_memory_local.py
```

运行后，脚本会提示：
```
Please enter your DeepSeek API key:
```

输入你的 DeepSeek API 密钥并按回车。注意：为了保密，输入时屏幕上不会显示你输入的内容。

**第7步：退出虚拟环境（运行完毕后）**
```bash
deactivate
```

## Troubleshooting

### SSL Error with tiktoken

If you encounter `SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING]` when running the script, this is because `tiktoken` needs to download tokenizer data from OpenAI's servers. This is a known issue on some macOS systems with SSL certificate restrictions.

**Solution: Use the local embedding version**

Instead of `conversational_rag_with_memory.py`, use `conversational_rag_with_memory_local.py` which uses local HuggingFace embeddings:

```bash
# Install sentence-transformers
pip install sentence-transformers

# Run the local version
python conversational_rag_with_memory_local.py
```

The local version works exactly the same but doesn't require downloading external tokenizer data.

```