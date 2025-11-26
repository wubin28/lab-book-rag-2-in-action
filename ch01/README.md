## Quickstart for macOS iter2 zsh

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
```bash
pip install langchain openai faiss-cpu tiktoken
```

**第5步：准备DeepSeek API密钥**

这个脚本使用 DeepSeek-reasoner API。在运行时，脚本会提示你输入 DeepSeek API 密钥。
请确保你已经注册并获取了 DeepSeek API 密钥（访问 https://platform.deepseek.com）。

**第6步：运行脚本**
```bash
python conversational_rag_with_memory.py
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

```