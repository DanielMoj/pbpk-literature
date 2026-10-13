# improved_triage.py - Verbesserte Triage-Klasse mit Scan-Erkennung
import json
import pymupdf as fitz
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from datetime import datetime

from document_pair import DocumentPair
from resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class AdvancedScanDetector:
    """Advanced scan detection using multiple heuristics."""
    
    def __init__(self, threshold: float = 0.6):
        """
        Initialize scan detector.
        
        Args:
            threshold: Score threshold for scan detection (0-1)
        """
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
    
    def detect_scan(self, doc: fitz.Document) -> bool:
        """
        Detect if document is a scan using advanced heuristics.
        
        Args:
            doc: The PDF document to analyze
            
        Returns:
            bool: True if document appears to be a scan
        """
        try:
            features = self._extract_features(doc)
            scan_score = self._calculate_scan_score(features)
            
            is_scan = scan_score >= self.threshold
            self.logger.debug(f"Scan detection score: {scan_score:.3f}, threshold: {self.threshold}, is_scan: {is_scan}")
            return is_scan
            
        except Exception as e:
            self.logger.warning(f"Error in scan detection, defaulting to False: {e}")
            return False
    
    def _extract_features(self, doc: fitz.Document) -> Dict[str, float]:
        """Extract features for scan detection."""
        features = {
            'avg_text_density': 0.0,
            'avg_font_size': 0.0,
            'image_to_page_ratio': 0.0,
            'font_diversity': 0.0,
            'text_line_density': 0.0,
            'large_image_coverage': 0.0,
            'has_selectable_text': False,
            'uniform_font_size': False
        }
        
        if doc.page_count == 0:
            return features
        
        # Smart sampling: first, middle, last pages
        sample_pages = min(3, doc.page_count)
        if doc.page_count <= 3:
            sample_indices = list(range(doc.page_count))
        else:
            sample_indices = [0, doc.page_count // 2, doc.page_count - 1]
        
        total_text_chars = 0
        total_page_area = 0
        all_font_sizes = []
        total_images = 0
        total_text_lines = 0
        large_image_area = 0
        has_selectable_text = False
        
        for i in sample_indices:
            try:
                page = doc[i]
                page_area = page.rect.width * page.rect.height
                if page_area <= 0:
                    continue
                    
                total_page_area += page_area
                
                # Text analysis
                text = page.get_text().strip()
                if text:
                    has_selectable_text = True
                    total_text_chars += len(text)
                    total_text_lines += len([line for line in text.split('\n') if line.strip()])
                
                # Font analysis
                font_sizes = self._get_font_sizes(page)
                all_font_sizes.extend(font_sizes)
                
                # Image analysis
                images = page.get_images()
                total_images += len(images)
                
                # Large image coverage analysis
                for img_ref in images:
                    try:
                        bbox = page.get_image_bbox(img_ref)
                        img_area = bbox.width * bbox.height
                        coverage = img_area / page_area
                        
                        # Consider images covering >50% of page as "large"
                        if coverage > 0.5:
                            large_image_area += img_area
                    except Exception:
                        continue
                        
            except Exception as e:
                self.logger.debug(f"Error analyzing page {i}: {e}")
                continue
        
        # Calculate normalized features
        if sample_pages > 0 and total_page_area > 0:
            features['avg_text_density'] = total_text_chars / total_page_area
            features['image_to_page_ratio'] = total_images / sample_pages
            features['text_line_density'] = total_text_lines / sample_pages
            features['large_image_coverage'] = large_image_area / total_page_area
            features['has_selectable_text'] = has_selectable_text
        
        if all_font_sizes:
            features['avg_font_size'] = np.mean(all_font_sizes)
            features['font_diversity'] = len(set(all_font_sizes)) / len(all_font_sizes)
            # Check if font sizes are very uniform (typical for OCR)
            features['uniform_font_size'] = np.std(all_font_sizes) < 1.0
        
        return features
    
    def _get_font_sizes(self, page: fitz.Page) -> List[float]:
        """Extract font sizes from page."""
        font_sizes = []
        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size = span.get("size", 0)
                            if size > 0:
                                font_sizes.append(size)
        except Exception as e:
            self.logger.debug(f"Error extracting font sizes: {e}")
        
        return font_sizes
    
    def _calculate_scan_score(self, features: Dict[str, float]) -> float:
        """Calculate scan probability score based on features."""
        score = 0.0
        
        # No selectable text is strong indicator of scan
        if not features['has_selectable_text']:
            score += 0.4
        
        # Very low text density indicates scan
        if features['avg_text_density'] < 0.01:
            score += 0.3
        elif features['avg_text_density'] < 0.05:
            score += 0.2
        
        # High image-to-page ratio indicates scan
        if features['image_to_page_ratio'] >= 1.0:
            score += 0.2
        elif features['image_to_page_ratio'] >= 0.5:
            score += 0.1
        
        # Large image coverage indicates scan
        if features['large_image_coverage'] > 0.7:
            score += 0.3
        elif features['large_image_coverage'] > 0.4:
            score += 0.2
        
        # Uniform font size (typical for OCR)
        if features['uniform_font_size'] and features['avg_font_size'] > 0:
            score += 0.1
        
        # Low font diversity
        if features['font_diversity'] < 0.3 and features['avg_font_size'] > 0:
            score += 0.1
        
        # Very low text line density
        if features['text_line_density'] < 5:
            score += 0.1
        
        return min(score, 1.0)


class ImprovedDocumentTriageSystem:
    """Improved document triage system with batch processing and validation."""
    
    def __init__(self, config, resource_manager: ResourceManager):
        """
        Initialize document triage system.
        
        Args:
            config: Configuration object
            resource_manager: Resource manager for memory management
        """
        self.pdf_dir = Path(config.paths.pdf_dir)
        self.metadata_dir = Path(config.paths.metadata_dir)
        self.resource_manager = resource_manager
        self.scan_detector = AdvancedScanDetector(threshold=0.6)
        self.logger = logging.getLogger(__name__)
        
        # Validate directories
        if not self.pdf_dir.exists():
            raise ValueError(f"PDF directory not found: {self.pdf_dir}")
        if not self.metadata_dir.exists():
            raise ValueError(f"Metadata directory not found: {self.metadata_dir}")
    
    def find_document_pairs(self) -> List[DocumentPair]:
        """Find all PDF-JSON document pairs in the configured directories."""
        pairs = []

        # Detect structure: folder-based (doi_xxx/paper.pdf) vs flat (*.pdf + *.json)
        # any() erzwingt Auswertung des Generators — ohne das wäre jede Bedingung truthy.
        folder_based = any(self.pdf_dir.glob("*/paper.pdf"))

        if folder_based:
            # Structure: pdf_dir/doi_xxx/{paper.pdf, metadata.json}
            self.logger.info(f"Using folder-based structure in {self.pdf_dir}")

            # Find all subdirectories containing paper.pdf
            paper_folders = [f.parent for f in self.pdf_dir.glob("*/paper.pdf")]
            self.logger.info(f"Found {len(paper_folders)} paper folders in {self.pdf_dir}")

            for folder in paper_folders:
                pdf_path = folder / "paper.pdf"
                json_path = folder / "metadata.json"

                # Create document pair
                pair = DocumentPair(
                    pdf_path=pdf_path,
                    json_path=json_path
                )

                pairs.append(pair)

        else:
            # Flat structure: separate pdf and metadata directories
            pdf_files = list(self.pdf_dir.glob("*.pdf"))
            self.logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_dir}")

            for pdf_path in pdf_files:
                # Find corresponding JSON metadata file
                json_name = pdf_path.stem + ".json"
                json_path = self.metadata_dir / json_name

                # Create document pair
                pair = DocumentPair(
                    pdf_path=pdf_path,
                    json_path=json_path
                )

                pairs.append(pair)

        return pairs
    
    def _validate_and_classify_pair(self, pair: DocumentPair) -> DocumentPair:
        """Validate and classify a single document pair."""
        if not pair.is_valid:
            return pair
        
        try:
            # Validate PDF structure
            valid = pair.validate_pdf_structure(self.resource_manager)
            if not valid:
                return pair
            
            # Detect if document is a scan
            with self.resource_manager.managed_document(pair.pdf_path) as doc:
                pair.is_scan = self.scan_detector.detect_scan(doc)
                
                # Additional validation
                if doc.is_encrypted:
                    pair.is_valid = False
                    pair.error_message = "PDF is encrypted"
                elif doc.needs_pass:
                    pair.is_valid = False
                    pair.error_message = "PDF requires password"
                elif doc.is_repaired:
                    self.logger.warning(f"PDF was repaired: {pair.pdf_path.name}")
            
            self.logger.debug(
                f"Validated {pair.pdf_path.name}: "
                f"pages={pair.page_count}, scan={pair.is_scan}, "
                f"size={pair.file_size_mb:.1f}MB"
            )
            
        except Exception as e:
            pair.is_valid = False
            pair.error_message = f"Validation error: {str(e)}"
            self.logger.error(f"Error validating {pair.pdf_path.name}: {e}")
        
        return pair
    
    def process_batch(self, max_workers: int = 4) -> Tuple[List[DocumentPair], List[DocumentPair]]:
        """
        Process all document pairs in parallel.
        
        Args:
            max_workers: Maximum number of parallel workers
            
        Returns:
            Tuple of (valid_pairs, invalid_pairs)
        """
        self.logger.info("Starting document triage...")
        
        # Find all document pairs
        all_pairs = self.find_document_pairs()
        
        if not all_pairs:
            self.logger.warning("No document pairs found")
            return [], []
        
        valid_pairs = []
        invalid_pairs = []
        
        # Process pairs in parallel
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='TriageWorker') as executor:
            # Submit all validation tasks
            future_to_pair = {
                executor.submit(self._validate_and_classify_pair, pair): pair
                for pair in all_pairs
            }
            
            # Process results with progress bar
            with tqdm(total=len(all_pairs), desc="Stage 1: Document Triage") as pbar:
                for future in as_completed(future_to_pair):
                    try:
                        validated_pair = future.result()
                        
                        if validated_pair.is_valid:
                            valid_pairs.append(validated_pair)
                        else:
                            invalid_pairs.append(validated_pair)
                            self.logger.warning(
                                f"Invalid document: {validated_pair.pdf_path.name} - "
                                f"{validated_pair.error_message}"
                            )
                    
                    except Exception as e:
                        original_pair = future_to_pair[future]
                        original_pair.is_valid = False
                        original_pair.error_message = f"Processing error: {str(e)}"
                        invalid_pairs.append(original_pair)
                        self.logger.error(f"Error processing {original_pair.pdf_path.name}: {e}")
                    
                    finally:
                        pbar.update(1)
        
        # Sort pairs by file size for better processing order
        valid_pairs.sort(key=lambda p: p.file_size_mb)
        
        # Log statistics
        self._log_triage_statistics(valid_pairs, invalid_pairs)
        
        return valid_pairs, invalid_pairs
    
    def _log_triage_statistics(self, valid_pairs: List[DocumentPair], invalid_pairs: List[DocumentPair]):
        """Log statistics about the triage results."""
        total_pairs = len(valid_pairs) + len(invalid_pairs)
        
        if total_pairs == 0:
            return
        
        # Calculate statistics
        scan_count = sum(1 for p in valid_pairs if p.is_scan)
        native_count = len(valid_pairs) - scan_count
        total_size_mb = sum(p.file_size_mb for p in valid_pairs)
        avg_size_mb = total_size_mb / len(valid_pairs) if valid_pairs else 0
        
        self.logger.info("=" * 60)
        self.logger.info("TRIAGE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total documents found: {total_pairs}")
        self.logger.info(f"Valid documents: {len(valid_pairs)} ({len(valid_pairs)/total_pairs*100:.1f}%)")
        self.logger.info(f"  - Native PDFs: {native_count}")
        self.logger.info(f"  - Scanned PDFs: {scan_count}")
        self.logger.info(f"Invalid documents: {len(invalid_pairs)} ({len(invalid_pairs)/total_pairs*100:.1f}%)")
        self.logger.info(f"Total size: {total_size_mb:.1f}MB (avg: {avg_size_mb:.1f}MB)")
        
        if invalid_pairs:
            # Group errors
            error_counts = {}
            for pair in invalid_pairs:
                error_type = pair.error_message.split(":")[0] if pair.error_message else "Unknown"
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            self.logger.info("Invalid document reasons:")
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  - {error_type}: {count}")
        
        self.logger.info("=" * 60)
    
    def save_triage_report(self, valid_pairs: List[DocumentPair], invalid_pairs: List[DocumentPair], output_path: Path):
        """Save detailed triage report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_documents": len(valid_pairs) + len(invalid_pairs),
                "valid_documents": len(valid_pairs),
                "invalid_documents": len(invalid_pairs),
                "scan_documents": sum(1 for p in valid_pairs if p.is_scan),
                "native_documents": sum(1 for p in valid_pairs if not p.is_scan)
            },
            "valid_documents": [p.get_summary() for p in valid_pairs],
            "invalid_documents": [p.get_summary() for p in invalid_pairs]
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Triage report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Could not save triage report: {e}")