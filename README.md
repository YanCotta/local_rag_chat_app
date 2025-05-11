# Local RAG Chat Assistant with Gemma

A powerful, privacy-focused conversational AI that runs completely locally using Google's Gemma model and RAG (Retrieval-Augmented Generation) technology. This application combines modern document retrieval with local LLM capabilities to provide accurate, context-aware responses while maintaining data privacy.


## Features

- 🔒 **100% Local Processing**: All operations run on your machine
- 🤖 **Gemma Integration**: Powered by Google's Gemma model via Ollama
- 🔍 **Advanced RAG**: FAISS-based vector search for accurate context retrieval
- ⚡ **Modern UI**: Built with Reflex for a responsive experience
- 🔄 **Streaming Responses**: Real-time answer generation
- 📝 **Code Highlighting**: Automatic syntax highlighting for code blocks
- 💾 **Chat Export**: Export conversations to JSON
- ⚙️ **Configurable**: Adjustable temperature and streaming settings

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional, for better performance)

## Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/local_rag_chat_app.git
   cd local_rag_chat_app
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install Gemma Model**
   ```bash
   ollama pull gemma3:4b-it-qat
   ```

4. **Launch Application**
   ```bash
   reflex run
   ```

5. **Access the Interface**
   - Open http://localhost:3000 in your browser

## Configuration

Create a `.env` file in the project root:

```env
OLLAMA_MODEL=gemma3:4b-it-qat
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_PATH=./vector_store
DATASET_SUBSET_SIZE=100
```

## Project Structure

```
local_rag_chat_app/
├── rag_gemma_reflex/
│   ├── components.py    # UI components
│   ├── state.py        # Application state management
│   ├── styles.py       # UI styling
│   ├── rag_logic.py    # RAG implementation
│   └── error_handling.py # Error management
├── vector_store/       # FAISS index storage
└── requirements.txt    # Dependencies
```

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running (`ollama serve`)
- **Memory Errors**: Try reducing `DATASET_SUBSET_SIZE` in `.env`
- **Slow Responses**: Consider using a GPU or reducing context window size

## Development

- Uses modern Python async/await patterns
- Implements error boundary pattern for robustness
- Supports hot-reloading during development
- Includes comprehensive error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) for details

---

Built with 🚀 using Reflex, LangChain, and Ollama
