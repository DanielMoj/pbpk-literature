# document_pair.py - Document Pair Datenklasse mit Validierung
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any
import hashlib
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class DocumentPair:
    """
    Represents a PDF-JSON document pair with validation and metadata.
    """
    pdf_path: Path
    json_path: Path
    pdf_hash: str = ""
    is_scan: bool = False
    is_valid: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_size_mb: float = 0.0
    page_count: int = 0
    
    def __post_init__(self):
        """Validate and enrich document pair after initialization."""
        # Ensure paths are Path objects
        self.pdf_path = Path(self.pdf_path)
        self.json_path = Path(self.json_path)
        
        # Validate PDF exists
        if not self.pdf_path.exists():
            self.is_valid = False
            self.error_message = f"PDF file not found: {self.pdf_path}"
            logger.warning(self.error_message)
            return
            
        # Validate JSON exists
        if not self.json_path.exists():
            self.is_valid = False
            self.error_message = f"Metadata file not found: {self.json_path}"
            logger.warning(self.error_message)
            return
        
        # Calculate file size
        try:
            self.file_size_mb = self.pdf_path.stat().st_size / 1024 / 1024
        except Exception as e:
            logger.warning(f"Could not get file size for {self.pdf_path}: {e}")
        
        # Calculate hash if not provided
        if not self.pdf_hash:
            self.pdf_hash = self._calculate_file_hash()
        
        # Try to load metadata
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                if not isinstance(self.metadata, dict):
                    logger.warning(f"Metadata is not a dictionary for {self.json_path}")
                    self.metadata = {"raw": self.metadata}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.json_path}: {e}")
            self.metadata = {"error": f"JSON decode error: {str(e)}"}
        except Exception as e:
            logger.error(f"Could not load metadata from {self.json_path}: {e}")
            self.metadata = {"error": str(e)}
    
    def _calculate_file_hash(self) -> str:
        """Calculate SHA256 hash for the PDF file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(self.pdf_path, "rb") as f:
                # Read in chunks for better memory efficiency
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Could not calculate hash for {self.pdf_path}: {e}")
            return ""
    
    def validate_pdf_structure(self, resource_manager) -> bool:
        """
        Validate PDF structure and update metadata.
        
        Args:
            resource_manager: ResourceManager instance for memory management
            
        Returns:
            bool: True if PDF is valid
        """
        if not self.is_valid:
            return False
            
        try:
            with resource_manager.managed_document(self.pdf_path) as doc:
                self.page_count = doc.page_count
                
                if self.page_count == 0:
                    self.is_valid = False
                    self.error_message = "PDF has no pages"
                    return False
                
                # Basic structure validation
                try:
                    # Try to access first page
                    first_page = doc[0]
                    # Try to get text - this will fail for severely corrupted PDFs
                    _ = first_page.get_text()
                    
                    # Check if PDF is encrypted
                    if doc.is_encrypted:
                        self.is_valid = False
                        self.error_message = "PDF is encrypted"
                        return False
                        
                    # Check if PDF has restrictions
                    if doc.permissions < 0:
                        logger.warning(f"PDF has restrictions: {self.pdf_path}")
                        
                except Exception as e:
                    self.is_valid = False
                    self.error_message = f"PDF structure validation failed: {e}"
                    return False
                
            return True
            
        except Exception as e:
            self.is_valid = False
            self.error_message = f"PDF validation error: {e}"
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the document pair."""
        return {
            "pdf_path": str(self.pdf_path),
            "json_path": str(self.json_path),
            "pdf_hash": self.pdf_hash,
            "is_scan": self.is_scan,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "file_size_mb": round(self.file_size_mb, 2),
            "page_count": self.page_count,
            "has_metadata": bool(self.metadata and "error" not in self.metadata)
        }
    
    def __repr__(self) -> str:
        """String representation of DocumentPair."""
        return (f"DocumentPair(pdf='{self.pdf_path.name}', "
                f"valid={self.is_valid}, scan={self.is_scan}, "
                f"pages={self.page_count}, size={self.file_size_mb:.1f}MB)")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "valid" if self.is_valid else f"invalid ({self.error_message})"
        doc_type = "scan" if self.is_scan else "native"
        return f"{self.pdf_path.name} - {status}, {doc_type}, {self.page_count} pages, {self.file_size_mb:.1f}MB"