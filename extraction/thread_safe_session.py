# thread_safe_session.py - Thread-sichere Session-Verwaltung
import threading
import requests
from typing import Optional, Dict, Any
import logging
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)

class ThreadSafeSessionManager:
    """Manages thread-local HTTP sessions for concurrent processing."""
    
    def __init__(self, 
                 timeout: int = 300,
                 max_retries: int = 3,
                 pool_connections: int = 10,
                 pool_maxsize: int = 10,
                 backoff_factor: float = 0.3):
        """
        Initialize thread-safe session manager.
        
        Args:
            timeout: Default timeout for requests in seconds
            max_retries: Maximum number of retries
            pool_connections: Number of connection pools to cache
            pool_maxsize: Maximum number of connections to save in the pool
            backoff_factor: Backoff factor for retries
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize
        self.backoff_factor = backoff_factor
        self._local = threading.local()
        self._lock = threading.Lock()
        self._session_count = 0
    
    @property
    def session(self) -> requests.Session:
        """Get or create a thread-local HTTP session."""
        if not hasattr(self._local, 'session') or self._local.session is None:
            self._local.session = self._create_session()
            with self._lock:
                self._session_count += 1
            logger.debug(f"Created new session for thread {threading.current_thread().name} (total: {self._session_count})")
        return self._local.session
    
    def _create_session(self) -> requests.Session:
        """Create a new configured HTTP session."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize,
            max_retries=retry_strategy
        )
        
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': 'PDF-Extraction-Pipeline/2.0',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        return session
    
    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make a thread-safe HTTP request.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
        """
        # Set default timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
            
        return self.session.request(method, url, **kwargs)
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a GET request."""
        return self.request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """Make a POST request."""
        return self.request('POST', url, **kwargs)
    
    def close_session(self):
        """Close the current thread's session."""
        if hasattr(self._local, 'session') and self._local.session is not None:
            try:
                self._local.session.close()
                with self._lock:
                    self._session_count -= 1
                logger.debug(f"Closed session for thread {threading.current_thread().name} (remaining: {self._session_count})")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            finally:
                self._local.session = None
    
    def close_all_sessions(self):
        """Close all sessions (call from main thread after all work is done)."""
        # Note: This only closes the current thread's session
        # Other threads need to call close_session() themselves
        self.close_session()
        logger.info(f"Session manager cleanup completed. Active sessions: {self._session_count}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        return {
            "active_sessions": self._session_count,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "pool_connections": self.pool_connections,
            "pool_maxsize": self.pool_maxsize
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.close_session()

# Convenience functions for common use cases
def create_session(timeout: int = 300, max_retries: int = 3) -> requests.Session:
    """
    Create a standalone configured session.
    
    Args:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        
    Returns:
        Configured requests session
    """
    manager = ThreadSafeSessionManager(timeout=timeout, max_retries=max_retries)
    return manager._create_session()

def make_request_with_retry(url: str, 
                          method: str = 'GET',
                          timeout: int = 300,
                          max_retries: int = 3,
                          **kwargs) -> Optional[requests.Response]:
    """
    Make a single request with retry logic.
    
    Args:
        url: Request URL
        method: HTTP method
        timeout: Request timeout
        max_retries: Maximum retries
        **kwargs: Additional request arguments
        
    Returns:
        Response object or None on failure
    """
    try:
        with ThreadSafeSessionManager(timeout=timeout, max_retries=max_retries) as manager:
            response = manager.request(method, url, **kwargs)
            response.raise_for_status()
            return response
    except requests.RequestException as e:
        logger.error(f"Request failed after {max_retries} retries: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in request: {e}")
        return None