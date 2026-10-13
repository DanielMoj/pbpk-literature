# retry_utils.py - Retry-Mechanismus mit Exponential Backoff
import functools
import time
import random
import logging
from typing import Callable, Type, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Retry decorator with exponential backoff and jitter.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay after each failure
        jitter: Add random jitter to prevent thundering herd
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function that will retry on specified exceptions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                        
                        # Add jitter to prevent thundering herd
                        if jitter:
                            delay += random.uniform(0, delay * 0.1)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        break
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
            
            raise last_exception
        return wrapper
    return decorator

def retry_on_failure(max_retries: int = 3, 
                    delay: float = 1.0, 
                    backoff: float = 2.0):
    """
    Simple retry decorator (backward compatibility).
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        
    Returns:
        Decorator function
    """
    return retry_with_exponential_backoff(
        max_retries=max_retries,
        base_delay=delay,
        backoff_factor=backoff,
        jitter=True,
        exceptions=(Exception,)
    )

class RetryableError(Exception):
    """Base exception for retryable errors."""
    pass

class NonRetryableError(Exception):
    """Base exception for non-retryable errors."""
    pass

def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.
    
    Args:
        error: The exception to check
        
    Returns:
        bool: True if the error is retryable
    """
    # Network-related errors are typically retryable
    retryable_error_types = (
        ConnectionError,
        TimeoutError,
        RetryableError,
    )
    
    # Check if it's explicitly non-retryable
    if isinstance(error, NonRetryableError):
        return False
    
    # Check if it's a known retryable type
    if isinstance(error, retryable_error_types):
        return True
    
    # Check error messages for common retryable scenarios
    error_msg = str(error).lower()
    retryable_keywords = [
        'timeout', 'timed out',
        'connection', 'network',
        'temporary', 'unavailable',
        '502', '503', '504',  # HTTP status codes
        'rate limit', 'too many requests'
    ]
    
    return any(keyword in error_msg for keyword in retryable_keywords)

def retry_with_callback(
    max_retries: int = 3,
    base_delay: float = 1.0,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    should_retry: Optional[Callable[[Exception], bool]] = None
):
    """
    Advanced retry decorator with callbacks.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        on_retry: Callback function called on each retry (error, attempt_number)
        should_retry: Function to determine if error should trigger retry
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry this error
                    if should_retry and not should_retry(e):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise
                    
                    if attempt < max_retries:
                        # Call retry callback if provided
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        # Calculate delay
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Retrying {func.__name__} after {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator