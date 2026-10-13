# improved_google_ai.py - Verbesserte Google Document AI Verarbeitung
from __future__ import annotations

# Google Document AI ist optional (pip install google-cloud-documentai).
# Ohne das Paket bleibt das Modul importierbar; erst der Prozessor meldet den Fehler.
try:
    from google.cloud import documentai_v1 as documentai
    from google.api_core.client_options import ClientOptions
    from google.api_core import exceptions as google_exceptions
    _GOOGLE_RETRY_ERRORS = (
        google_exceptions.GoogleAPIError,
        google_exceptions.RetryError,
        google_exceptions.DeadlineExceeded,
    )
except ImportError:
    documentai = None
    _GOOGLE_RETRY_ERRORS = ()

from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import pymupdf as fitz
import time

from retry_utils import retry_with_exponential_backoff

logger = logging.getLogger(__name__)

class ImprovedGoogleDocumentAIProcessor:
    """Improved Google Document AI processor with better error handling and validation."""
    
    def __init__(self, config):
        """
        Initialize Google Document AI processor.
        
        Args:
            config: Configuration object with Google Cloud settings
        """
        if documentai is None:
            raise ImportError(
                "Google Document AI ist optional und nicht installiert: "
                "pip install google-cloud-documentai"
            )

        self.project_id = config.google.project_id
        self.location = config.google.location
        self.table_processor_id = config.google.table_processor_id
        self.ocr_processor_id = config.google.ocr_processor_id
        self.timeout = config.google.timeout
        self.max_retries = config.google.max_retries
        self.logger = logging.getLogger(__name__)
        
        # Check if Google Cloud credentials are configured
        self._validate_credentials()
        
        # Initialize client
        try:
            opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
            self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
            
            # Build processor paths
            self.table_processor_name = self.client.processor_path(
                self.project_id, self.location, self.table_processor_id
            )
            self.ocr_processor_name = self.client.processor_path(
                self.project_id, self.location, self.ocr_processor_id
            )
            
            self.logger.info("Google Document AI client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Document AI client: {e}")
            raise
    
    def _validate_credentials(self):
        """Validate that Google Cloud credentials are properly configured."""
        if self.project_id == "your-google-project-id" or self.project_id.startswith("<"):
            self.logger.warning(
                "Google Cloud project ID not configured. "
                "Please update the configuration with your actual project ID."
            )
            raise ValueError("Google Cloud project ID not configured")
        
        # Check for credentials in environment
        import os
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            self.logger.warning(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable not set. "
                "Please set it to point to your service account key file."
            )
    
    def _get_text_from_layout(self, layout: documentai.Document.Page.Layout, text: str) -> str:
        """Extract text based on layout segments with error handling."""
        if not layout or not layout.text_anchor or not layout.text_anchor.text_segments:
            return ""
        
        response = ""
        try:
            for segment in layout.text_anchor.text_segments:
                start_index = int(segment.start_index) if segment.start_index else 0
                end_index = int(segment.end_index) if segment.end_index else len(text)
                
                # Validate indices
                if 0 <= start_index <= end_index <= len(text):
                    response += text[start_index:end_index]
                else:
                    self.logger.warning(
                        f"Invalid text segment indices: {start_index}-{end_index} "
                        f"for text length {len(text)}"
                    )
        except Exception as e:
            self.logger.warning(f"Error extracting text from layout: {e}")
        
        return response.strip()
    
    def _parse_table_result(self, document: documentai.Document) -> List[Dict[str, Any]]:
        """Parse table results with enhanced validation."""
        if not document or not document.pages:
            self.logger.warning("No pages found in Document AI response")
            return []
        
        all_tables_data = []
        
        for page_idx, page in enumerate(document.pages):
            if not hasattr(page, 'tables') or not page.tables:
                continue
            
            for table_idx, table in enumerate(page.tables):
                try:
                    table_data = {
                        "page": page_idx + 1,
                        "table_index": table_idx + 1,
                        "rows": [],
                        "confidence": float(table.layout.confidence) if table.layout and hasattr(table.layout, 'confidence') else 0.0,
                        "row_count": 0,
                        "col_count": 0,
                        "has_headers": False
                    }
                    
                    max_cols = 0
                    
                    # Process header rows
                    if hasattr(table, 'header_rows') and table.header_rows:
                        table_data["has_headers"] = True
                        for row_idx, row in enumerate(table.header_rows):
                            row_data = self._parse_table_row(row, document.text, is_header=True)
                            table_data["rows"].append({
                                "row_type": "header",
                                "row_index": row_idx,
                                "cells": row_data
                            })
                            max_cols = max(max_cols, len(row_data))
                    
                    # Process body rows
                    if hasattr(table, 'body_rows') and table.body_rows:
                        for row_idx, row in enumerate(table.body_rows):
                            row_data = self._parse_table_row(row, document.text, is_header=False)
                            table_data["rows"].append({
                                "row_type": "body",
                                "row_index": row_idx,
                                "cells": row_data
                            })
                            max_cols = max(max_cols, len(row_data))
                    
                    table_data["row_count"] = len(table_data["rows"])
                    table_data["col_count"] = max_cols
                    
                    if table_data["rows"]:  # Only add tables with content
                        all_tables_data.append(table_data)
                        
                except Exception as e:
                    self.logger.error(f"Error parsing table on page {page_idx + 1}: {e}")
                    continue
        
        self.logger.debug(f"Parsed {len(all_tables_data)} tables from Document AI response")
        return all_tables_data
    
    def _parse_table_row(self, row, document_text: str, is_header: bool = False) -> List[Dict[str, Any]]:
        """Parse a single table row with error handling."""
        row_data = []
        
        if not hasattr(row, 'cells') or not row.cells:
            return row_data
        
        for cell_idx, cell in enumerate(row.cells):
            cell_data = {
                "text": "",
                "is_header": is_header,
                "col_span": 1,
                "row_span": 1,
                "confidence": 0.0,
                "cell_index": cell_idx
            }
            
            try:
                # Extract cell text
                if hasattr(cell, 'layout') and cell.layout:
                    cell_data["text"] = self._get_text_from_layout(cell.layout, document_text)
                    if hasattr(cell.layout, 'confidence'):
                        cell_data["confidence"] = float(cell.layout.confidence)
                
                # Extract span information
                if hasattr(cell, 'col_span') and cell.col_span:
                    cell_data["col_span"] = int(cell.col_span)
                if hasattr(cell, 'row_span') and cell.row_span:
                    cell_data["row_span"] = int(cell.row_span)
                    
            except Exception as e:
                self.logger.warning(f"Error parsing table cell: {e}")
            
            row_data.append(cell_data)
        
        return row_data
    
    @retry_with_exponential_backoff(
        max_retries=3,
        exceptions=_GOOGLE_RETRY_ERRORS
    )
    def extract_table_from_page(self, pdf_path: Path, page_num: int) -> Optional[List[Dict[str, Any]]]:
        """
        Extract tables from a specific PDF page with validation.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (1-based)
            
        Returns:
            List of extracted tables or None on failure
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        self.logger.info(f"Extracting tables from {pdf_path.name}, page {page_num}")
        
        try:
            # Read PDF content
            with open(pdf_path, 'rb') as f:
                content = f.read()
            
            if not content:
                self.logger.error(f"Empty PDF file: {pdf_path}")
                return None
            
            # Check file size
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > 10:  # 10MB limit for Document AI
                self.logger.warning(f"Large PDF file ({file_size_mb:.1f}MB), may exceed API limits")
            
            # Prepare request
            raw_document = documentai.RawDocument(
                content=content, 
                mime_type="application/pdf"
            )
            
            # Create process options for specific page if processor supports it
            process_options = documentai.ProcessOptions()
            # Note: Page-specific processing might not be supported by all processors
            
            # Field mask to only get table information
            field_mask = documentai.FieldMask(paths=["pages.tables", "text"])
            
            request = documentai.ProcessRequest(
                name=self.table_processor_name,
                raw_document=raw_document,
                field_mask=field_mask,
                process_options=process_options
            )
            
            # Process document
            result = self.client.process_document(request=request, timeout=self.timeout)
            
            # Validate response
            if not result or not result.document:
                self.logger.error("Empty response from Google Document AI")
                return None
            
            if not result.document.pages:
                self.logger.warning("No pages found in Document AI response")
                return []
            
            # Filter for target page (DocumentAI is 0-based, our input is 1-based)
            target_page_index = page_num - 1
            
            if target_page_index >= len(result.document.pages):
                self.logger.warning(
                    f"Page {page_num} not found in Document AI response "
                    f"(only {len(result.document.pages)} pages)"
                )
                return []
            
            # Create document with only the target page
            target_page = result.document.pages[target_page_index]
            filtered_document = documentai.Document(
                pages=[target_page],
                text=result.document.text
            )
            
            tables = self._parse_table_result(filtered_document)
            self.logger.info(f"Extracted {len(tables)} tables from {pdf_path.name}, page {page_num}")
            return tables
            
        except google_exceptions.InvalidArgument as e:
            self.logger.error(f"Invalid argument error for {pdf_path.name}: {e}")
            return None
        except google_exceptions.ResourceExhausted as e:
            self.logger.error(f"Quota exceeded for {pdf_path.name}: {e}")
            raise  # Let retry mechanism handle this
        except google_exceptions.DeadlineExceeded as e:
            self.logger.error(f"Request timeout for {pdf_path.name}: {e}")
            raise  # Let retry mechanism handle this
        except google_exceptions.GoogleAPIError as e:
            self.logger.error(f"Google API error for {pdf_path.name}, page {page_num}: {e}")
            raise  # Let retry mechanism handle this
        except Exception as e:
            self.logger.error(f"Unexpected error in table extraction for {pdf_path.name}, page {page_num}: {e}")
            return None
    
    def _extract_page_structure(self, document: documentai.Document) -> List[Dict[str, Any]]:
        """Extract page structure from OCR result."""
        pages_data = []
        
        if not document or not document.pages:
            return pages_data
        
        for i, page in enumerate(document.pages):
            page_data = {
                "page_number": i + 1,
                "paragraphs": [],
                "tables": [],
                "confidence": 0.0,
                "detected_languages": []
            }
            
            # Extract detected languages
            if hasattr(page, 'detected_languages') and page.detected_languages:
                page_data["detected_languages"] = [
                    {
                        "language_code": lang.language_code,
                        "confidence": float(lang.confidence) if hasattr(lang, 'confidence') else 0.0
                    }
                    for lang in page.detected_languages
                ]
            
            # Extract paragraphs
            if hasattr(page, 'paragraphs') and page.paragraphs:
                paragraph_texts = []
                confidences = []
                
                for para in page.paragraphs:
                    if para.layout:
                        para_text = self._get_text_from_layout(para.layout, document.text)
                        if para_text.strip():
                            paragraph_texts.append(para_text)
                            if hasattr(para.layout, 'confidence'):
                                confidences.append(float(para.layout.confidence))
                
                page_data["paragraphs"] = paragraph_texts
                page_data["confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Extract tables if present
            if hasattr(page, 'tables') and page.tables:
                page_tables = self._parse_table_result(
                    documentai.Document(pages=[page], text=document.text)
                )
                page_data["tables"] = page_tables
            
            pages_data.append(page_data)
        
        return pages_data
    
    @retry_with_exponential_backoff(
        max_retries=3,
        exceptions=_GOOGLE_RETRY_ERRORS
    )
    def process_scanned_pdf(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a scanned PDF with OCR to extract text and structure.
        
        Args:
            pdf_path: Path to the scanned PDF file
            
        Returns:
            Dict containing OCR results or None on failure
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        self.logger.info(f"Processing scanned PDF with OCR: {pdf_path.name}")
        
        try:
            # Read PDF content
            with open(pdf_path, 'rb') as f:
                content = f.read()
            
            if not content:
                self.logger.error(f"Empty PDF file: {pdf_path}")
                return None
            
            # Check file size
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > 10:  # 10MB limit for Document AI
                self.logger.warning(f"Large PDF file ({file_size_mb:.1f}MB), may exceed API limits")
            
            # Prepare request
            raw_document = documentai.RawDocument(
                content=content,
                mime_type="application/pdf"
            )
            
            # OCR specific field mask
            field_mask = documentai.FieldMask(
                paths=[
                    "text",
                    "pages.layout",
                    "pages.paragraphs",
                    "pages.lines",
                    "pages.tokens",
                    "pages.tables",
                    "pages.detected_languages",
                    "pages.blocks"
                ]
            )
            
            request = documentai.ProcessRequest(
                name=self.ocr_processor_name,
                raw_document=raw_document,
                field_mask=field_mask
            )
            
            # Process document with OCR
            start_time = time.time()
            result = self.client.process_document(request=request, timeout=self.timeout)
            processing_time = time.time() - start_time
            
            self.logger.info(f"OCR processing completed in {processing_time:.1f}s")
            
            # Validate response
            if not result or not result.document:
                self.logger.error("Empty response from Google Document AI OCR")
                return None
            
            document = result.document
            
            # Extract structured data
            ocr_data = {
                "full_text": document.text,
                "pages": self._extract_page_structure(document),
                "confidence": 0.0,
                "processing_time": processing_time,
                "page_count": len(document.pages) if document.pages else 0
            }
            
            # Calculate overall confidence
            page_confidences = [
                page.get("confidence", 0.0) 
                for page in ocr_data["pages"] 
                if page.get("confidence", 0.0) > 0
            ]
            
            if page_confidences:
                ocr_data["confidence"] = sum(page_confidences) / len(page_confidences)
            
            # Extract tables across all pages
            all_tables = []
            for page_data in ocr_data["pages"]:
                if page_data.get("tables"):
                    all_tables.extend(page_data["tables"])
            
            ocr_data["total_tables"] = len(all_tables)
            
            # Extract detected languages
            all_languages = {}
            for page_data in ocr_data["pages"]:
                for lang_info in page_data.get("detected_languages", []):
                    lang_code = lang_info["language_code"]
                    confidence = lang_info["confidence"]
                    if lang_code in all_languages:
                        all_languages[lang_code].append(confidence)
                    else:
                        all_languages[lang_code] = [confidence]
            
            # Average confidence per language
            ocr_data["detected_languages"] = [
                {
                    "language_code": lang,
                    "confidence": sum(confs) / len(confs)
                }
                for lang, confs in all_languages.items()
            ]
            
            self.logger.info(
                f"OCR completed for {pdf_path.name}: "
                f"{ocr_data['page_count']} pages, "
                f"{len(ocr_data['full_text'])} characters, "
                f"confidence: {ocr_data['confidence']:.2%}"
            )
            
            return ocr_data
            
        except google_exceptions.InvalidArgument as e:
            self.logger.error(f"Invalid argument error for OCR {pdf_path.name}: {e}")
            return None
        except google_exceptions.ResourceExhausted as e:
            self.logger.error(f"Quota exceeded for OCR {pdf_path.name}: {e}")
            raise  # Let retry mechanism handle this
        except google_exceptions.DeadlineExceeded as e:
            self.logger.error(f"OCR timeout for {pdf_path.name}: {e}")
            raise  # Let retry mechanism handle this
        except google_exceptions.GoogleAPIError as e:
            self.logger.error(f"Google API error for OCR {pdf_path.name}: {e}")
            raise  # Let retry mechanism handle this
        except Exception as e:
            self.logger.error(f"Unexpected error in OCR for {pdf_path.name}: {e}")
            return None
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about configured processors."""
        info = {
            "project_id": self.project_id,
            "location": self.location,
            "processors": {
                "table": {
                    "id": self.table_processor_id,
                    "name": self.table_processor_name
                },
                "ocr": {
                    "id": self.ocr_processor_id,
                    "name": self.ocr_processor_name
                }
            }
        }
        
        # Try to get processor details
        try:
            # Note: This would require additional API calls to get processor details
            # For now, just return the basic info
            pass
        except Exception as e:
            self.logger.warning(f"Could not get processor details: {e}")
        
        return info