# Privacy-First RAG Chat Assistant

A locally-hosted conversational AI application leveraging Retrieval-Augmented Generation (RAG) technology. Built with cutting-edge open-source tools, this application ensures complete data privacy while providing intelligent responses powered by context-aware document retrieval.

![RAG Architecture](assets/rag_diagram.png)

## Core Technologies

- 🔒 100% Local Processing - No cloud dependencies
- 🤖 Ollama for LLM hosting
- 🔍 FAISS for efficient vector search
- ⚡ Reflex for the web interface
- 🔗 LangChain for RAG orchestration
- 🤗 Hugging Face for embeddings

## Setup Requirements

Before running the application, ensure you have:

- Python 3.12 or newer
- Ollama installed locally (https://ollama.com)
- The Gemma language model: `ollama pull gemma3:4b-it-qat`

## Quick Start

1. Get the code:

   ```bash
   git clone https://github.com/YanCotta/local_rag_chat_app.git
   cd local_rag_chat_app
   ```

2. Set up dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:

   ```bash
   reflex run
   ```

4. Visit http://localhost:3000 in your browser

## Architecture Overview

The application follows a modular architecture:

```
local_rag_chat_app/
├── vector_store/       # FAISS index storage
├── app/               # Core application code
│   ├── ui.py         # Frontend components
│   ├── rag.py        # RAG implementation
│   └── config.py     # Settings
└── data/             # Document storage
```

## How RAG Works Here

1. Document Processing:
   - Text documents are converted to vector embeddings
   - Vectors are stored in a FAISS database

2. Question Processing:
   - User questions are vectorized
   - Similar documents are retrieved
   - Context-enhanced responses are generated

## Configuration

Key settings can be adjusted through environment variables:

- `MODEL_NAME`: Choose your Ollama model
- `EMBEDDING_MODEL`: Select embedding model
- `VECTOR_STORE_PATH`: Database location

## Credits

This project builds upon these amazing tools:
- LangChain
- Reflex Framework
- Ollama
- FAISS
- Hugging Face

## License

Released under MIT License

---

Built with ❤️ for privacy-focused AI applications
