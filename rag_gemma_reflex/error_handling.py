"""Error handling utilities for the RAG chat application."""
import asyncio
from typing import TypeVar, Callable, Any, Optional, Tuple
from functools import wraps
import traceback

T = TypeVar('T')

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

async def with_retries(
    func: Callable[..., T],
    *args,
    retry_config: RetryConfig = None,
    on_retry: Callable[[int, Exception], None] = None,
    **kwargs,
) -> Tuple[Optional[T], bool]:
    """
    Execute a function with retry logic.
    
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

    while attempt < config.max_retries:
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result, True

        except config.exceptions as e:
            attempt += 1
            if attempt < config.max_retries:
                if on_retry:
                    on_retry(attempt, e)
                await asyncio.sleep(current_delay)
                current_delay *= config.backoff_factor
            else:
                print(f"Maximum retry attempts ({config.max_retries}) reached")
                print(f"Last error: {str(e)}")
                print(traceback.format_exc())
                return None, False

    return None, False

def format_error_message(error: Exception, user_friendly: bool = True) -> str:
    """Format error message for display."""
    if user_friendly:
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
    """Context manager for handling errors in async code."""
    
    def __init__(self, on_error: Callable[[Exception], Any] = None):
        self.on_error = on_error
        self.error = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.error = exc_val
            if self.on_error:
                await self.on_error(exc_val)
            return True  # Suppress the exception
        return False
