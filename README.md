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

## Development and Testing

### Running Tests

The project uses pytest for testing. To run the test suite:

```bash
# Install test dependencies
pip install -r requirements.txt

# Run tests with coverage
pytest

# Generate coverage report
pytest --cov=rag_gemma_reflex --cov-report=html
```

### Development Guidelines

1. **Code Style**
   - Use type hints consistently
   - Follow PEP 8 guidelines
   - Document all public functions and classes

2. **Error Handling**
   - Use custom exception classes from `error_handling.py`
   - Always log errors appropriately
   - Provide user-friendly error messages

3. **Performance**
   - Use batch processing for large operations
   - Implement proper cleanup for memory management
   - Monitor and optimize resource usage

4. **Testing**
   - Write unit tests for new features
   - Maintain test coverage above 80%
   - Add integration tests for critical paths

### Monitoring and Debugging

The application includes comprehensive logging:

```python
import logging
logger = logging.getLogger(__name__)

# Logs are available in rag_chat.log
logging.basicConfig(
    filename='rag_chat.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Performance Optimization

- FAISS index is cached between sessions
- Batch processing for document embeddings
- Memory cleanup after heavy operations
- Rate limiting to prevent overload

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) for details

---

Built with 🚀 using Reflex, LangChain, and Ollama
