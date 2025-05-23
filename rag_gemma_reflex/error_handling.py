"""Error handling utilities for the RAG chat application."""
import asyncio
from typing import TypeVar, Callable, Any, Optional, Tuple, Dict
from functools import wraps
import traceback
from datetime import datetime
import logging
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class RAGError(Exception):
    """Base exception class for RAG-related errors."""
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __post_init__(self):
        super().__init__(self.message)
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


class ModelConnectionError(RAGError):
    """Raised when there are issues connecting to the LLM."""
    pass


class DatasetError(RAGError):
    """Raised when there are issues with the dataset."""
    pass


class VectorStoreError(RAGError):
    """Raised when there are issues with the vector store."""
    pass


class RateLimitError(RAGError):
    """Raised when rate limits are exceeded."""
    pass


class ValidationError(RAGError):
    """Raised when input validation fails."""
    pass


class RetryConfig:
    """Configuration for retry behavior."""
    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 2.0,
        backoff_factor: float = 1.5,
        exceptions: tuple = (Exception,),
    ):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log an error with context."""
    error_dict = {
        "error_type": error.__class__.__name__,
        "message": str(error),
        "context": context or {}
    }
    
    if isinstance(error, RAGError):
        error_dict.update(error.to_dict())
    
    logger.error(
        f"Error occurred: {error_dict['error_type']}", 
        extra={"error_details": error_dict}
    )


async def with_retries(
    func: Callable[..., T],
    *args,
    retry_config: RetryConfig = None,
    on_retry: Callable[[int, Exception], None] = None,
    **kwargs,
) -> Tuple[Optional[T], bool]:
    """
    Execute a function with retry logic and proper error handling.
    
    Args:
        func: The function to execute
        *args: Function arguments
        retry_config: Retry configuration
        on_retry: Callback for retry events
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, success)
    """
    config = retry_config or RetryConfig()
    attempt = 0
    current_delay = config.delay
    last_error = None

    while attempt < config.max_retries:
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result, True

        except config.exceptions as e:
            last_error = e
            attempt += 1
            log_error(e, {
                "attempt": attempt,
                "function": func.__name__,
                "args": args,
                "kwargs": kwargs
            })
            
            if attempt < config.max_retries:
                if on_retry:
                    on_retry(attempt, e)
                await asyncio.sleep(current_delay)
                current_delay *= config.backoff_factor
            else:
                logger.error(
                    f"Maximum retry attempts ({config.max_retries}) reached",
                    extra={"last_error": str(last_error)}
                )
                return None, False

    return None, False


def format_error_message(error: Exception, user_friendly: bool = True) -> str:
    """Format error message for display with improved user feedback."""
    if user_friendly:
        if isinstance(error, ModelConnectionError):
            return (
                "I'm having trouble connecting to the AI model. This might be because:\n\n"
                "1. The Ollama server is not running\n"
                "2. The model hasn't been downloaded yet\n"
                "3. There are network connectivity issues\n\n"
                "Please check that Ollama is running and try again."
            )
        elif isinstance(error, RateLimitError):
            return (
                "You've reached the rate limit for requests. Please wait a moment "
                "before trying again to ensure smooth operation of the system."
            )
        elif isinstance(error, ValidationError):
            return str(error)
        else:
            return (
                "I apologize, but I encountered an error. This might be due to:\n\n"
                "1. Connection issues with the language model\n"
                "2. Problems processing your query\n"
                "3. System resource constraints\n\n"
                "Please try again in a moment."
            )
    else:
        return f"Error: {str(error)}\n{traceback.format_exc()}"


class ErrorBoundary:
    """Context manager for handling errors in async code with improved logging."""
    
    def __init__(self, on_error: Callable[[Exception], Any] = None):
        self.on_error = on_error
        self.error = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.error = exc_val
            log_error(exc_val)
            if self.on_error:
                await self.on_error(exc_val)
            return True  # Suppress the exception
        return False


def validate_input(func: Callable) -> Callable:
    """Decorator for input validation."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Add input validation logic here
            return await func(*args, **kwargs)
        except Exception as e:
            log_error(e, {"function": func.__name__, "args": args, "kwargs": kwargs})
            raise ValidationError(str(e))
    return wrapper
