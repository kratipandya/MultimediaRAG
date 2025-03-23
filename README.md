# Project Schedule


| First Week    | Second Week   | Third Week   | Fourth Week |
| ------------- | ------------- |------------- |-------------|
| ✅Planning    | ✅Model + UI Skeletons | ✅Integration |  ✅Polishing   |
# Project 17: Multimedia Large Language Model Applications with Multimedia Embedding RAG

## Basic Idea
RAG models primarily focus on text-based retrieval, limiting their effectiveness in multimedia
applications. This project enhances RAG with multimodal retrieval by incorporating image, video, and
text embeddings. By integrating image and text embeddings in queries, the system enables faster and
more efficient retrieval, even on low-bandwidth networks. The use of compact embeddings ensures
quick access to relevant multimedia content, making AI-generated responses more contextually rich
and diverse

Create a scientific research assistant that retrieves related text, images, and videos from academic
papers.

## Requirements & Installation [Justfile](https://github.com/casey/just)
Node.js \
Python version: `Python 3.12`
```
pip install faiss-cpu torch transformers sentence-transformers langchain langchain_huggingface pypdf pydantic pillow robyn colorlog
```

**Get faiss-embeddings from HuggingFace:**
```bash
python ./retrieve-faiss.py
```
**Install node-packages**
```
just install
# cd frontend/react-web | pnpm install
```
**Test system is working with zero-shot**
```bash
just zero-shot
# cd backend/RAGembedder | py -3.12 .\zero_shot.py
```

**Start Frontend & Backend:**
```bash
just run-front
# cd frontend | pnpm run
```
```bash
just run-back
# cd backend | py -3.12 .\api.py
```
# Main files of the System
Contains all the information handling the faiss queries.
```
multi_modal_embedder.py
```
Handles the LLM powered with RAG generation
```
arxiv_rag_system.py
```
Contains all the logic behind backend and handling the `faiss-embedding class` and `rag-system calls`.
```
api_queue.py
```

# Technologies and Tecniques used
## Frontend
 - Main framework `React.js` with Typescript, `Rsbuild` with `Tailwindcss`.

Tecniques used
1. [SVG -noise](https://css-tricks.com/grainy-gradients/) for more natural looking UI.
2. [Material UI]().
3. [Tailwindcss]().


## Backend
Tecniques and Matrial arts used
1. [HTTP caching]() -for fast request retrival.
2. [RAG-retrival]() caching.
3. [Multithreading]() & [Hash-Cache]() for even faster faiss-retrival.
