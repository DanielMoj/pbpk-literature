# resource_manager.py - Ressourcenverwaltung und Memory Management
import gc
import psutil
import threading
import pymupdf as fitz
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages system resources and memory usage during PDF processing."""
    
    def __init__(self, max_memory_mb: int = 2048, memory_check_interval: int = 10):
        """
        Initialize resource manager.
        
        Args:
            max_memory_mb: Maximum memory usage in megabytes
            memory_check_interval: Check memory every N documents
        """
        self.max_memory_mb = max_memory_mb
        self.memory_check_interval = memory_check_interval
        self._document_counter = 0
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._memory_stats = []
        
    @contextmanager
    def managed_document(self, pdf_path: Path) -> Generator[fitz.Document, None, None]:
        """
        Context manager for PDF documents with automatic memory management.
        
        Args:
            pdf_path: Path to the PDF file
            
        Yields:
            fitz.Document: The opened PDF document
            
        Raises:
            MemoryError: If memory usage exceeds limits
            FileNotFoundError: If PDF file doesn't exist
        """
        doc = None
        
        # Ensure path is a Path object
        pdf_path = Path(pdf_path)
        
        with self._lock:
            self._document_counter += 1
            doc_count = self._document_counter
        
        try:
            # Check if file exists
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Check memory before opening document
            self._check_memory_usage(f"before opening document {pdf_path.name}")
            
            # Open document
            doc = fitz.open(pdf_path)
            logger.debug(f"Opened document {pdf_path.name} (#{doc_count})")
            
            # Yield document for processing
            yield doc
            
        except fitz.FileDataError as e:
            logger.error(f"Invalid PDF file {pdf_path.name}: {e}")
            raise ValueError(f"Cannot open PDF file: {pdf_path.name} - {e}")
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error processing document {pdf_path.name}: {e}")
            raise
        finally:
            # Always close document and clean up
            if doc:
                try:
                    doc.close()
                    logger.debug(f"Closed document {pdf_path.name} (#{doc_count})")
                except Exception as e:
                    logger.warning(f"Error closing document {pdf_path.name}: {e}")
            
            # Periodic garbage collection
            if doc_count % self.memory_check_interval == 0:
                gc.collect()
                self._check_memory_usage(f"after processing {doc_count} documents")
    
    def _check_memory_usage(self, context: str = ""):
        """Check current memory usage and log warnings if high."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            # Record memory statistics
            self._memory_stats.append({
                'timestamp': time.time() - self._start_time,
                'memory_mb': memory_mb,
                'memory_percent': memory_percent,
                'context': context
            })
            
            log_msg = f"Memory usage {context}: {memory_mb:.1f}MB ({memory_percent:.1f}%)"
            
            if memory_mb > self.max_memory_mb:
                logger.warning(f"{log_msg} - EXCEEDS LIMIT ({self.max_memory_mb}MB)")
                # Force garbage collection
                gc.collect()
                
                # Check again after GC
                new_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                if new_memory_mb > self.max_memory_mb:
                    raise MemoryError(
                        f"Memory usage ({new_memory_mb:.1f}MB) exceeds limit ({self.max_memory_mb}MB) "
                        f"even after garbage collection"
                    )
                else:
                    logger.info(f"Memory reduced to {new_memory_mb:.1f}MB after garbage collection")
            else:
                logger.debug(log_msg)
                
        except psutil.NoSuchProcess:
            logger.warning("Cannot check memory usage - process not found")
        except Exception as e:
            logger.warning(f"Error checking memory usage: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Calculate peak memory from recorded stats
            peak_memory_mb = max((stat['memory_mb'] for stat in self._memory_stats), default=0)
            
            return {
                "current_rss_mb": memory_info.rss / 1024 / 1024,
                "current_vms_mb": memory_info.vms / 1024 / 1024,
                "current_percent": process.memory_percent(),
                "peak_memory_mb": peak_memory_mb,
                "documents_processed": self._document_counter,
                "uptime_seconds": time.time() - self._start_time
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total_mb": memory.total / 1024 / 1024,
                    "available_mb": memory.available / 1024 / 1024,
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / 1024 / 1024 / 1024,
                    "free_gb": disk.free / 1024 / 1024 / 1024,
                    "percent": disk.percent
                }
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup."""
        logger.info("Forcing memory cleanup...")
        gc.collect()
        
        # Get memory stats after cleanup
        stats = self.get_memory_stats()
        if stats:
            logger.info(f"Memory after cleanup: {stats['current_rss_mb']:.1f}MB")

class LazyDocumentLoader:
    """
    Lazy loading wrapper for PDF documents.
    
    This class provides lazy loading of PDF documents to optimize memory usage.
    """
    
    def __init__(self, pdf_path: Path):
        """
        Initialize lazy document loader.
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        self._doc = None
        
    @property
    def doc(self) -> fitz.Document:
        """Get the document, loading it if necessary."""
        if self._doc is None:
            if not self.pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            self._doc = fitz.open(self.pdf_path)
        return self._doc
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure document is closed."""
        if self._doc:
            try:
                self._doc.close()
            except Exception as e:
                logger.warning(f"Error closing document: {e}")
            finally:
                self._doc = None
    
    def close(self):
        """Explicitly close the document."""
        if self._doc:
            self._doc.close()
            self._doc = None