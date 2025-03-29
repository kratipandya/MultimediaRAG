
# Project 17: Multimedia Large Language Model Applications with Multimedia Embedding RAG

![gif](preview/preview.gif)

## Abstract
This project implements a multimodal Retrieval-Augmented Generation (RAG) search engine focused on scientific content from arXiv. The system allows users to search through research papers using text queries, image uploads, or audio inputs, and provides AI-generated answers based on relevant content. The architecture combines a React frontend with a Python backend using FAISS vector search and the DeepSeek R1 language model to deliver accurate, context-aware responses along with supporting scientific papers, images, audio files, and videos.


# 1. Introduction
## 1.1 Background
The exponential growth of scientific literature makes it increasingly difficult for researchers to find relevant papers and extract valuable insights efficiently. Traditional keyword-based search engines often fail to capture semantic relationships between concepts, while standard large language models lack up-to-date scientific knowledge and citation capabilities.

RAG models primarily focus on text-based retrieval, limiting their effectiveness in multimedia applications. This project enhances RAG with multimodal retrieval by incorporating image, video, and text embeddings. By integrating image and text embeddings in queries, the system enables faster and more efficient retrieval. With even on low-bandwidth networks. The use of **compact embeddings** ensures quick access to **relevant multimedia content**, making AI-generated responses more contextually rich and diverse.


## 1.2 Installation Instructions
The following **Instructions** where tested in Windows 11 enviroment.

Make sure these are installed in the system, if not already:
1. [Node.js](https://nodejs.org/en)
2. [pnpm](https://pnpm.io/installation) or try with `npm install -g pnpm`
3. [Python 3.12.9](https://www.python.org/downloads/) tested in the 3.12.19 version
4. Recommended [Justfile](https://github.com/casey/just) -not mandatory

*If your system does not hae python and node installed, we recommended that you reboot your system.*

## Setup with `just`

Using `just` you can **install**, **build**, and retrive **faiss embeddings** using command:
```just
just setup-all
```

## Setup without `just`
**Install backend dependencies**
```bash
py -3.12 -m pip install -r requirements.txt
```
**Retrieve FAISS indices from HuggingFace**
```bash
py -3.12 .\retrieve-faiss.py
```
**Install node-packages**
```
cd frontend
pnpm install
```

# Usage guide
## Test zero-shot
**Test that the FAISS retrieving is working using zero-shot**
```bash
just zero-shot
# commnad runs: cd backend/rag | py -3.12 .\zero_shot.py
```

## Running
**Start Frontend & Backend in separate terminals**
```bash
just run-back
# commnad runs: cd backend | py -3.12 .\api.py
```
```bash
just
just run-front
# commnad runs: cd frontend | pnpm run dev
```

Starting the frontend, open's up a browser to interact with the system!


## 1.2 Objectives
Creating **Scientific Research Assistant** that retrieves related text, images, and videos from academic
papers.
- Develop a multimodal search interface ( that accepts text, image, and audio inputs)
- Vector search across in related multimedia content
- Generate accurate, contextual answers using retrieved documents
- Present supporting evidence alongside responses for verification and further exploration
- Create an intuitive user experience for scientific information retrieval


# 2. System Overview
### 2.1 System Description
The ArXiv RAG Search Engine consists of two main components:

1. **Frontend**: A React application providing a clean interface for submitting queries, displaying results, and interacting with retrieved content. It supports text search input along with image and audio file uploads.

2. **Backend**: A Python service handling query processing, vector embedding, similarity search, and answer generation. It uses a multi-threaded architecture to process queries asynchronously.

The system follows a RAG workflow:
- User submits a query (text, image, and/or audio)
- Query is embedded into vector representations
- Similar content is retrieved from multiple FAISS indices
- Retrieved content is used to create a prompt for the LLM
- LLM generates a comprehensive answer with citations
- Results are displayed along with supporting evidence

### 2.2 System Features
- **Multimodal Input**: Support for text queries, PNG image uploads, and WAV audio uploads
- **Vector Search**: FAISS-powered similarity search across multiple indices
- **AI-Generated Answers**: Contextual responses using **DeepSeek R1 Distill Qwen 1.5B**
- **Paper References**: Relevant arXiv papers with similarity scores
- **Multimedia Results**: Related images, audio clips, and YouTube videos
- **Interactive UI**: Modern interface with loading animations and content previews

### 2.3 System Sctructure
```
MultiModal-ResearchAssistant/
├── backend/
│   ├── __init__.py
│   ├── api.py                  # Main API entry point
│   ├── api_dirty.py            # Helper functions for API
│   ├── api_queue.py            # Query handling and queue management
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── arxiv_rag_system.py # RAG implementation
│   │   ├── build.py            # Build embeddings
│   │   ├── multi_modal_embedder.py # Embedder for multi-modal content
│   │   ├── prompt.py           # Prompt templates
│   │   └── zero_shot.py        # Zero-shot testing
│   └── RAGembedder/            # FAISS indices (downloaded)
│
├── frontend/
│   ├── index.html              # HTML entry point
│   ├── ....
│   └── src/
│       ├── App.tsx             # Main App component
│       ├── index.jsx           # React entry point
│       ├── assets/*
│       ├── components/*        # App.tsx components
│       ├── pages/
│       │   ├── HomePage.tsx
│       │   ├── ResultPage.tsx
│       │   ├── ...
│       │   ├── homepage/*
│       │   └── resultpage/*
│       ├── services/
│       │   └── RagApi.tsx
│       └── styles/
│           └── index.css
│
├── .gitignore
├── justfile                    # Task runner commands
├── requirements.txt            # Python dependencies
└── retrieve-faiss.py           # Script to download FAISS indices
```

# 3. Tools and Technologies Used

**Frontend:**
- `React` with `TypeScript` for application framework
- `Material UI` and `Tailwind CSS` for component styling
- `React Router` for navigation

**Backend:**
- `Python 3.12` for server-side logic
- `Robyn` for API server
- `PyTorch` for machine learning operations
- `Transformers` for model implementations
- `FAISS` for vector similarity search
- `OpenCV` for image processing
- `Librosa` for audio processing
- `PyTubeFix` and **YouTube Transcript API** for video content
- `MultiThreading` for **Multimodality search** and **Response Generation**
- **HTTP-cahching** with hashing for faster retrieval


###  Models
| Modality  | Model                            | Dimensionality |
| --------- | ----------------------------------------- | -------------- |
| Textual   | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)    | 384            |
| Audio     | [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en)                   | 384            |
| Image     | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)              | 512            |
| Video     | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)              | 512            |
| Answer Generation | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | |

**Textual-Image modalities** are cross-searched using multi-modal `CLIP` -model with projection layer.
### About - FAISS embeddings
 - **Textual** -embeddings contains around [2.7million arXiv research paper](https://www.kaggle.com/datasets/Cornell-University/arxiv) -collected 03/2025, indexed using paper's title & abstract -collected.
 - **Image & Sound** -embeddings contains indexed debug data.
 - **Video** -embeddings contains context from **3Blue1Brown** youtube channel, indexed using (title and transcription data) + (average index of **4fps** frames).

### 3.2 Implementation Details
The implementation follows a modular architecture:

**Multimodal Embedder:**
- Created a unified embedding system that handles text, images, audio, and video
- Implemented vector projections between different embedding spaces for cross-modal search
- Optimized embedding caching to improve performance

**RAG Implementation:**
- Designed a pipeline for retrieving relevant documents from multiple sources
- Developed context-aware prompting templates to guide LLM responses
- Implemented clean-up mechanisms to remove duplicate or low-quality results

**Frontend Components:**
- Created a responsive search interface with drag-and-drop file uploads
- Developed custom components for displaying different media types
- Implemented loading states with informative messages during processing

**API and Queue System:**
- Built an asynchronous processing queue to handle long-running operations
- Designed a caching system to store query results for faster retrieval
- Implemented error handling and graceful degradation


# Summary
The ArXiv RAG Search Engine successfully demonstrates how retrieval-augmented generation can enhance scientific information retrieval. By combining multimodal inputs, efficient vector search, and contextual answer generation, the system provides a powerful tool for exploring research papers and related content. The implementation proves that modern AI techniques can be applied to create more intuitive and informative scientific search experiences.


<!-- # Main files of the System
**FAISS** -contains all the handling and searching embeddings of indices:\
`multi_modal_embedder.py`

**LLM** -handles answer generation\
`arxiv_rag_system.py`

**Backend Handler** - Handles queue's request's from **frontend**\
`api_queue.py` -->



