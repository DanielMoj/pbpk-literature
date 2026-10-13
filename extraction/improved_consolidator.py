# improved_consolidator.py - Verbesserte Datenkonsolidierung
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from document_pair import DocumentPair
from resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class ImprovedDataConsolidator:
    """Improved data consolidator with better validation and error handling."""
    
    def __init__(self, config, resource_manager: ResourceManager):
        """
        Initialize data consolidator.
        
        Args:
            config: Configuration object
            resource_manager: Resource manager for memory management
        """
        self.output_dir = Path(config.paths.output_dir)
        self.image_dir = Path(config.paths.image_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(__name__)
        
        # Pipeline version
        self.pipeline_version = "2.0"
    
    def _determine_section_level(self, section: Dict[str, Any]) -> int:
        """Determine hierarchical level of section."""
        # Use existing level if available
        if "level" in section and isinstance(section.get("level"), int):
            return max(1, min(section["level"], 6))
        
        # Fallback heuristics based on title
        title = str(section.get("title", "")).strip().lower()
        if not title:
            return 1
        
        # Common main sections
        main_sections = ["abstract", "introduction", "conclusion", "references", "bibliography"]
        if any(word in title for word in main_sections):
            return 1
        
        # Numbered sections
        if title[0].isdigit():
            # Count dots to determine level
            dots = title.split()[0].count('.') if title.split() else 0
            return min(dots + 1, 6)
        
        # Default
        return 2
    
    def _calculate_text_statistics(self, text: str) -> Dict[str, int]:
        """Calculate text statistics."""
        if not text:
            return {"words": 0, "characters": 0, "sentences": 0}
        
        # Word count
        words = len(text.split())
        
        # Character count (excluding spaces)
        characters = len(text.replace(" ", ""))
        
        # Sentence count (simple heuristic)
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(1, sentences)  # At least 1 sentence if there's text
        
        return {
            "words": words,
            "characters": characters,
            "sentences": sentences
        }
    
    def _enhance_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance sections with IDs, hierarchy, and validation."""
        enhanced = []
        
        for i, section in enumerate(sections):
            if not isinstance(section, dict):
                self.logger.warning(f"Invalid section format at index {i}")
                continue
            
            enhanced_section = {
                "id": section.get("id", f"section_{i+1}"),
                "title": str(section.get("title", "")).strip(),
                "type": section.get("type", "section"),
                "level": self._determine_section_level(section),
                "elements": [],
                "statistics": {
                    "word_count": 0,
                    "character_count": 0,
                    "sentence_count": 0,
                    "element_count": 0,
                    "paragraph_count": 0,
                    "list_count": 0
                }
            }
            
            # Process elements
            elements = section.get("paragraphs", []) or section.get("elements", [])
            total_text = ""
            
            for j, element in enumerate(elements):
                if not isinstance(element, dict):
                    continue
                
                element_id = f"{enhanced_section['id']}_elem_{j+1}"
                element_type = element.get("type", "paragraph")
                
                enhanced_element = {
                    "id": element_id,
                    "type": element_type,
                    "coordinates": element.get("coordinates")
                }
                
                # Handle different element types
                if element_type == "paragraph":
                    text = str(element.get("text", "")).strip()
                    enhanced_element["text"] = text
                    enhanced_element["statistics"] = self._calculate_text_statistics(text)
                    total_text += " " + text
                    enhanced_section["statistics"]["paragraph_count"] += 1
                    
                elif element_type == "list":
                    items = element.get("items", [])
                    enhanced_element["items"] = [str(item).strip() for item in items if item]
                    enhanced_element["item_count"] = len(enhanced_element["items"])
                    list_text = " ".join(enhanced_element["items"])
                    enhanced_element["statistics"] = self._calculate_text_statistics(list_text)
                    total_text += " " + list_text
                    enhanced_section["statistics"]["list_count"] += 1
                    
                else:
                    # Generic handling for other types
                    for key, value in element.items():
                        if key not in ["type", "coordinates"]:
                            enhanced_element[key] = value
                
                enhanced_section["elements"].append(enhanced_element)
            
            # Calculate section statistics
            section_stats = self._calculate_text_statistics(total_text)
            enhanced_section["statistics"]["word_count"] = section_stats["words"]
            enhanced_section["statistics"]["character_count"] = section_stats["characters"]
            enhanced_section["statistics"]["sentence_count"] = section_stats["sentences"]
            enhanced_section["statistics"]["element_count"] = len(enhanced_section["elements"])
            
            # Only add sections with meaningful content
            if enhanced_section["title"] or enhanced_section["elements"]:
                enhanced.append(enhanced_section)
        
        return enhanced
    
    def _structure_ocr_content(self, ocr_data: Dict[str, Any], grobid_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Structure OCR content with improved organization."""
        sections = []
        
        if not ocr_data or "pages" not in ocr_data:
            self.logger.warning("No OCR pages data available")
            return sections
        
        # Check if Grobid found any structure
        grobid_sections = grobid_data.get("sections", [])
        has_grobid_structure = len(grobid_sections) > 0
        
        pages = ocr_data.get("pages", [])
        
        if has_grobid_structure:
            # Try to align OCR content with Grobid structure
            self.logger.info("Aligning OCR content with Grobid structure")
            # This is complex and would need more sophisticated text matching
            # For now, fallback to page-based structure
        
        # Create sections based on pages
        for page in pages:
            page_num = page.get("page_number", 1)
            paragraphs = page.get("paragraphs", [])
            
            if not paragraphs:
                continue
            
            # Group paragraphs into a section
            section = {
                "id": f"ocr_page_{page_num}",
                "title": f"Page {page_num}",
                "type": "ocr_page",
                "level": 1,
                "elements": [],
                "statistics": {
                    "word_count": 0,
                    "character_count": 0,
                    "sentence_count": 0,
                    "element_count": 0,
                    "paragraph_count": 0,
                    "list_count": 0
                },
                "ocr_confidence": page.get("confidence", 0.0)
            }
            
            total_text = ""
            
            for i, para_text in enumerate(paragraphs):
                if not para_text or not str(para_text).strip():
                    continue
                
                para_stats = self._calculate_text_statistics(para_text)
                
                element = {
                    "id": f"ocr_page_{page_num}_para_{i+1}",
                    "type": "paragraph",
                    "text": str(para_text).strip(),
                    "statistics": para_stats,
                    "confidence": page.get("confidence", 0.0)
                }
                
                section["elements"].append(element)
                total_text += " " + para_text
                section["statistics"]["paragraph_count"] += 1
            
            # Update section statistics
            section_stats = self._calculate_text_statistics(total_text)
            section["statistics"]["word_count"] = section_stats["words"]
            section["statistics"]["character_count"] = section_stats["characters"]
            section["statistics"]["sentence_count"] = section_stats["sentences"]
            section["statistics"]["element_count"] = len(section["elements"])
            
            if section["elements"]:
                sections.append(section)
        
        self.logger.info(f"Structured OCR content into {len(sections)} sections")
        return sections

    def _integrate_tables(self, 
                         grobid_tables: List[Dict[str, Any]], 
                         extracted_tables: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Integrate Grobid metadata with extracted table data."""
        integrated = []
        
        # Create lookup for extracted tables by page
        extracted_by_page = {}
        for page_num, tables in extracted_tables.items():
            extracted_by_page[page_num] = tables
        
        for grobid_table in grobid_tables:
            if not isinstance(grobid_table, dict):
                continue
            
            coords = grobid_table.get("coordinates", {})
            if not coords or "page" not in coords:
                self.logger.warning("Grobid table without coordinates")
                continue
            
            page_num = coords["page"]
            
            table = {
                "id": grobid_table.get("id", f"table_{len(integrated)+1}"),
                "caption": str(grobid_table.get("caption", "")).strip(),
                "label": grobid_table.get("label", ""),
                "position": {
                    "page": page_num,
                    "bbox": coords
                },
                "content": None,
                "extraction_confidence": 0.0,
                "row_count": 0,
                "col_count": 0,
                "has_headers": False,
                "extraction_method": "none",
                "grobid_xml": grobid_table.get("content_xml", "")
            }
            
            # Find matching extracted tables
            page_tables = extracted_by_page.get(page_num, [])
            
            if page_tables:
                # For now, take the first table on the page
                # TODO: Implement better matching based on position
                best_table = None
                best_overlap = 0.0
                
                # Simple matching: use the first table
                # In production, you'd want to match based on bounding box overlap
                if page_tables:
                    best_table = page_tables[0]
                
                if best_table:
                    table["content"] = best_table
                    table["extraction_confidence"] = best_table.get("confidence", 0.0)
                    table["row_count"] = best_table.get("row_count", 0)
                    table["col_count"] = best_table.get("col_count", 0)
                    table["has_headers"] = best_table.get("has_headers", False)
                    table["extraction_method"] = "google_documentai"
            else:
                table["extraction_method"] = "grobid_only"
            
            integrated.append(table)
        
        # Add any extracted tables not matched with Grobid
        matched_pages = {t["position"]["page"] for t in integrated}
        for page_num, tables in extracted_by_page.items():
            if page_num not in matched_pages:
                for i, extracted_table in enumerate(tables):
                    table = {
                        "id": f"table_extracted_p{page_num}_{i+1}",
                        "caption": "",
                        "label": "",
                        "position": {
                            "page": page_num,
                            "bbox": None
                        },
                        "content": extracted_table,
                        "extraction_confidence": extracted_table.get("confidence", 0.0),
                        "row_count": extracted_table.get("row_count", 0),
                        "col_count": extracted_table.get("col_count", 0),
                        "has_headers": extracted_table.get("has_headers", False),
                        "extraction_method": "google_documentai_only",
                        "grobid_xml": ""
                    }
                    integrated.append(table)
        
        self.logger.debug(f"Integrated {len(integrated)} tables")
        return integrated
    
    def _save_raw_image(self, doc_name: str, figure_id: str, img_data: bytes) -> Path:
        """Save raw image data and return relative path."""
        doc_image_dir = self.image_dir / doc_name
        doc_image_dir.mkdir(exist_ok=True)
        
        output_path = doc_image_dir / f"{figure_id}.png"
        
        try:
            with open(output_path, "wb") as f:
                f.write(img_data)
            
            # Return path relative to output directory
            return output_path.relative_to(self.output_dir)
        except Exception as e:
            self.logger.error(f"Error saving raw image {figure_id}: {e}")
            raise
    
    def _integrate_figures(self, 
                          grobid_figures: List[Dict[str, Any]], 
                          extracted_figures: Dict[str, Dict[str, Any]], 
                          doc_name: str) -> List[Dict[str, Any]]:
        """Integrate figure metadata with image data."""
        integrated = []
        
        for grobid_fig in grobid_figures:
            if not isinstance(grobid_fig, dict):
                continue
            
            fig_id = grobid_fig.get("id", f"figure_{len(integrated)+1}")
            coords = grobid_fig.get("coordinates", {})
            
            figure = {
                "id": fig_id,
                "caption": str(grobid_fig.get("caption", "")).strip(),
                "label": grobid_fig.get("label", ""),
                "position": {
                    "page": coords.get("page", 0) if coords else 0,
                    "bbox": coords
                },
                "image": None,
                "extraction_success": False,
                "extraction_method": "none",
                "graphic_ref": grobid_fig.get("graphic_ref")
            }
            
            # Add extracted image data if available
            if fig_id in extracted_figures:
                img_data = extracted_figures[fig_id]
                
                try:
                    image_info = {
                        "format": img_data.get("format", "png"),
                        "dimensions": {
                            "width": img_data.get("width", 0),
                            "height": img_data.get("height", 0),
                            "dpi": img_data.get("dpi", 0)
                        },
                        "quality_score": img_data.get("quality_score", 0.0),
                        "extraction_type": img_data.get("type", "unknown"),
                        "size_mb": img_data.get("size_mb", 0.0)
                    }
                    
                    if img_data.get("type") == "rendered":
                        # Path already exists
                        image_path = Path(img_data["path"])
                        try:
                            image_info["path"] = str(image_path.relative_to(self.output_dir))
                        except ValueError:
                            # If path is not relative to output_dir, use absolute path
                            image_info["path"] = str(image_path)
                        figure["extraction_method"] = "rendered"
                    elif img_data.get("type") == "raw" and "data" in img_data:
                        # Save raw image data
                        img_path = self._save_raw_image(doc_name, fig_id, img_data["data"])
                        image_info["path"] = str(img_path)
                        figure["extraction_method"] = "raw_extracted"
                    
                    figure["image"] = image_info
                    figure["extraction_success"] = True
                    
                except Exception as e:
                    self.logger.error(f"Error processing figure {fig_id}: {e}")
                    figure["extraction_method"] = "failed"
                    figure["extraction_error"] = str(e)
            
            integrated.append(figure)
        
        # Log statistics
        successful = sum(1 for f in integrated if f['extraction_success'])
        self.logger.debug(
            f"Integrated {len(integrated)} figures "
            f"({successful} successfully extracted)"
        )
        
        return integrated

    def consolidate_document(self,
                           pair: DocumentPair,
                           grobid_data: Optional[Dict[str, Any]],
                           table_extractions: Dict[int, List[Dict[str, Any]]],
                           figure_extractions: Dict[str, Dict[str, Any]],
                           ocr_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Consolidate all data sources into unified structure.
        
        Args:
            pair: Document pair with metadata
            grobid_data: Grobid extraction results
            table_extractions: Extracted tables by page number
            figure_extractions: Extracted figures by ID
            ocr_data: OCR results for scanned documents
            
        Returns:
            Consolidated document structure
        """
        # Get document name from parent folder if using folder structure
        if pair.pdf_path.name == "paper.pdf":
            doc_name = pair.pdf_path.parent.name  # Use folder name (e.g., doi_001_10.1002_bcp.70019)
        else:
            doc_name = pair.pdf_path.stem  # Use filename without extension
        
        # Base document structure
        document = {
            "document_id": doc_name,
            "metadata": {
                "original_metadata": pair.metadata,
                "file_info": {
                    "pdf_path": str(pair.pdf_path),
                    "file_size_mb": round(pair.file_size_mb, 2),
                    "page_count": pair.page_count,
                    "is_scan": pair.is_scan,
                    "document_hash": pair.pdf_hash
                },
                "processing_info": {
                    "extraction_timestamp": datetime.now().isoformat(),
                    "pipeline_version": self.pipeline_version,
                    "extraction_methods": []
                }
            },
            "content": {
                "sections": [],
                "tables": [],
                "figures": [],
                "references": []
            },
            "statistics": {
                "total_sections": 0,
                "total_tables": 0,
                "total_figures": 0,
                "total_references": 0,
                "total_words": 0,
                "total_characters": 0,
                "total_sentences": 0,
                "extraction_success": True,
                "quality_metrics": {}
            }
        }
        
        # Track extraction methods used
        extraction_methods = []
        
        # Handle Grobid processing
        if not grobid_data:
            document["metadata"]["processing_info"]["grobid_error"] = "No Grobid data provided"
            document["statistics"]["extraction_success"] = False
            self.logger.error(f"No Grobid data for {doc_name}")
        elif "error" in grobid_data:
            document["metadata"]["processing_info"]["grobid_error"] = grobid_data.get("error", "Unknown error")
            document["statistics"]["extraction_success"] = False
            self.logger.error(f"Grobid processing failed for {doc_name}: {grobid_data.get('error')}")
        else:
            extraction_methods.append("grobid")
            
            # Extract Grobid metadata
            if "metadata" in grobid_data:
                document["metadata"]["document_metadata"] = grobid_data["metadata"]
        
        # Process content based on document type
        try:
            if pair.is_scan and ocr_data:
                extraction_methods.append("ocr")
                document["content"]["sections"] = self._structure_ocr_content(ocr_data, grobid_data or {})
                document["metadata"]["processing_info"]["ocr_confidence"] = ocr_data.get("confidence", 0.0)
                document["statistics"]["quality_metrics"]["ocr_confidence"] = ocr_data.get("confidence", 0.0)
            elif grobid_data and "sections" in grobid_data:
                document["content"]["sections"] = self._enhance_sections(grobid_data.get("sections", []))
            
            # Integrate tables
            if grobid_data and "tables" in grobid_data:
                document["content"]["tables"] = self._integrate_tables(
                    grobid_data.get("tables", []), 
                    table_extractions
                )
                if table_extractions:
                    extraction_methods.append("table_extraction")
            
            # Integrate figures
            if grobid_data and "figures" in grobid_data:
                document["content"]["figures"] = self._integrate_figures(
                    grobid_data.get("figures", []), 
                    figure_extractions, 
                    doc_name
                )
                if figure_extractions:
                    extraction_methods.append("figure_extraction")
            
            # Add references
            if grobid_data and "references" in grobid_data:
                document["content"]["references"] = grobid_data.get("references", [])
            
            # Calculate statistics
            stats = document["statistics"]
            stats["total_sections"] = len(document["content"]["sections"])
            stats["total_tables"] = len(document["content"]["tables"])
            stats["total_figures"] = len(document["content"]["figures"])
            stats["total_references"] = len(document["content"]["references"])
            
            # Calculate text statistics
            total_words = 0
            total_chars = 0
            total_sentences = 0
            
            for section in document["content"]["sections"]:
                section_stats = section.get("statistics", {})
                total_words += section_stats.get("word_count", 0)
                total_chars += section_stats.get("character_count", 0)
                total_sentences += section_stats.get("sentence_count", 0)
            
            stats["total_words"] = total_words
            stats["total_characters"] = total_chars
            stats["total_sentences"] = total_sentences
            
            # Quality metrics
            stats["quality_metrics"]["successful_table_extractions"] = sum(
                1 for t in document["content"]["tables"] 
                if t.get("extraction_method") not in ["none", "grobid_only"]
            )
            stats["quality_metrics"]["successful_figure_extractions"] = sum(
                1 for f in document["content"]["figures"] 
                if f.get("extraction_success")
            )
            stats["quality_metrics"]["table_extraction_rate"] = (
                stats["quality_metrics"]["successful_table_extractions"] / stats["total_tables"]
                if stats["total_tables"] > 0 else 0.0
            )
            stats["quality_metrics"]["figure_extraction_rate"] = (
                stats["quality_metrics"]["successful_figure_extractions"] / stats["total_figures"]
                if stats["total_figures"] > 0 else 0.0
            )
            
            # Update extraction methods
            document["metadata"]["processing_info"]["extraction_methods"] = extraction_methods
            
        except Exception as e:
            self.logger.error(f"Error consolidating document {doc_name}: {e}")
            document["statistics"]["extraction_success"] = False
            document["metadata"]["processing_info"]["consolidation_error"] = str(e)
        
        return document
    
    def save_consolidated_document(self, document: Dict[str, Any]) -> bool:
        """
        Save consolidated document with error handling.
        
        Args:
            document: Consolidated document structure
            
        Returns:
            bool: True if successful
        """
        doc_id = document["document_id"]
        output_path = self.output_dir / f"{doc_id}_extracted.json"
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create document-specific image directory if needed
            if any(f.get("extraction_success") for f in document["content"].get("figures", [])):
                doc_image_dir = self.image_dir / doc_id
                doc_image_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
            
            # Log summary
            stats = document.get("statistics", {})
            quality = stats.get("quality_metrics", {})
            
            self.logger.info(
                f"Saved {doc_id}: "
                f"{stats.get('total_sections', 0)} sections, "
                f"{stats.get('total_tables', 0)} tables "
                f"({quality.get('successful_table_extractions', 0)} extracted), "
                f"{stats.get('total_figures', 0)} figures "
                f"({quality.get('successful_figure_extractions', 0)} extracted), "
                f"{stats.get('total_words', 0)} words"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Could not save document {doc_id}: {e}")
            return False
    
    def generate_summary_report(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary report for multiple documents."""
        report = {
            "total_documents": len(documents),
            "successful_documents": sum(1 for d in documents if d["statistics"]["extraction_success"]),
            "total_sections": sum(d["statistics"]["total_sections"] for d in documents),
            "total_tables": sum(d["statistics"]["total_tables"] for d in documents),
            "total_figures": sum(d["statistics"]["total_figures"] for d in documents),
            "total_references": sum(d["statistics"]["total_references"] for d in documents),
            "total_words": sum(d["statistics"]["total_words"] for d in documents),
            "average_quality_metrics": {}
        }
        
        # Calculate average quality metrics
        quality_sums = {}
        quality_counts = {}
        
        for doc in documents:
            for metric, value in doc["statistics"].get("quality_metrics", {}).items():
                if isinstance(value, (int, float)):
                    quality_sums[metric] = quality_sums.get(metric, 0) + value
                    quality_counts[metric] = quality_counts.get(metric, 0) + 1
        
        for metric, total in quality_sums.items():
            count = quality_counts[metric]
            if count > 0:
                report["average_quality_metrics"][metric] = round(total / count, 3)
        
        return report