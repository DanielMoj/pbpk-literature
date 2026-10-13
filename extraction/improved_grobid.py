# improved_grobid.py - Verbesserte Grobid-Verarbeitung
import requests
from lxml import etree
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from thread_safe_session import ThreadSafeSessionManager
from retry_utils import retry_with_exponential_backoff

logger = logging.getLogger(__name__)

class ImprovedGrobidProcessor:
    """Improved Grobid processor with thread safety, retry mechanism, and better error handling."""
    
    def __init__(self, config):
        """
        Initialize Grobid processor.
        
        Args:
            config: Configuration object with Grobid settings
        """
        self.grobid_url = config.grobid.url.rstrip('/')
        self.timeout = config.grobid.timeout
        self.max_retries = config.grobid.max_retries
        self.consolidate_citations = config.grobid.consolidate_citations
        
        # Thread-safe session manager
        self.session_manager = ThreadSafeSessionManager(
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        
        # XML namespaces for TEI parsing
        self.namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
        self.logger = logging.getLogger(__name__)
        
        # Test connection to Grobid
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Grobid service."""
        try:
            response = self.session_manager.get(f"{self.grobid_url}/api/isalive", timeout=5)
            if response.status_code == 200:
                self.logger.info(f"Successfully connected to Grobid at {self.grobid_url}")
            else:
                self.logger.warning(f"Grobid service returned status {response.status_code}")
        except Exception as e:
            self.logger.warning(f"Could not connect to Grobid service: {e}")
    
    def _get_text_content(self, element) -> str:
        """Extract text content from TEI element recursively."""
        if element is None:
            return ""
        try:
            # Use lxml's efficient text extraction
            text = etree.tostring(element, method='text', encoding='unicode')
            return text.strip() if text else ""
        except Exception as e:
            self.logger.warning(f"Error extracting text content: {e}")
            return ""
    
    def _get_coordinates(self, element) -> Optional[Dict[str, Any]]:
        """
        Extract bounding box coordinates from element.
        
        Returns:
            Dict with coordinates or None if not available
        """
        coords = element.get('coords')
        if not coords:
            return None
        
        try:
            # Split multiple coordinate sets (take first one)
            coord_parts = coords.split(';')[0].split(',')
            
            if len(coord_parts) >= 5:
                page = int(coord_parts[0])
                x1 = float(coord_parts[1])
                y1 = float(coord_parts[2])
                w = float(coord_parts[3])
                h = float(coord_parts[4])
                
                return {
                    "page": page,
                    "x1": x1,
                    "y1": y1,
                    "w": w,
                    "h": h,
                    "x2": x1 + w,
                    "y2": y1 + h
                }
        except (ValueError, IndexError) as e:
            self.logger.debug(f"Invalid coordinates format '{coords}': {e}")
            return None
        
        return None
    
    def _extract_list(self, list_element) -> Dict[str, Any]:
        """Extract structured lists with better error handling."""
        try:
            items = []
            for item in list_element.findall('.//tei:item', self.namespaces):
                item_text = self._get_text_content(item)
                if item_text.strip():  # Only add non-empty items
                    items.append(item_text)
            
            return {
                "type": "list",
                "items": items,
                "coordinates": self._get_coordinates(list_element)
            }
        except Exception as e:
            self.logger.warning(f"Error extracting list: {e}")
            return {"type": "list", "items": [], "coordinates": None}
    
    def _extract_sections(self, body_element) -> List[Dict[str, Any]]:
        """Extract hierarchical text structure with improved parsing."""
        sections = []
        if body_element is None:
            self.logger.warning("No body element found in document")
            return sections
        
        try:
            # Process div elements (sections)
            divs = body_element.findall('.//tei:div', self.namespaces)
            
            for i, div in enumerate(divs):
                section = {
                    "id": f"section_{i+1}",
                    "type": div.get('type', 'section'),
                    "title": "",
                    "paragraphs": [],
                    "level": self._determine_section_level(div),
                    "n": div.get('n', '')  # Section number if available
                }
                
                # Extract section title
                head = div.find('./tei:head', self.namespaces)
                if head is not None:
                    section["title"] = self._get_text_content(head)
                
                # Extract paragraphs and lists directly under this div
                p_elements = div.findall('./tei:p', self.namespaces)
                list_elements = div.findall('./tei:list', self.namespaces)
                for elem in p_elements + list_elements:
                    if elem.tag.endswith('p'):
                        paragraph_text = self._get_text_content(elem)
                        if paragraph_text.strip():  # Only add non-empty paragraphs
                            section["paragraphs"].append({
                                "type": "paragraph",
                                "text": paragraph_text,
                                "coordinates": self._get_coordinates(elem)
                            })
                    elif elem.tag.endswith('list'):
                        list_data = self._extract_list(elem)
                        if list_data["items"]:  # Only add non-empty lists
                            section["paragraphs"].append(list_data)
                
                # Only add sections with content
                if section["title"] or section["paragraphs"]:
                    sections.append(section)
            
            # Fallback: if no structured divs found, extract all paragraphs from body
            if not sections:
                self.logger.info("No structured sections found, extracting all paragraphs from body")
                fallback_section = {
                    "id": "section_1",
                    "type": "body",
                    "title": "Document Body",
                    "paragraphs": [],
                    "level": 1
                }
                
                p_elements = body_element.findall('.//tei:p', self.namespaces)
                list_elements = body_element.findall('.//tei:list', self.namespaces)
                for elem in p_elements + list_elements:
                    if elem.tag.endswith('p'):
                        paragraph_text = self._get_text_content(elem)
                        if paragraph_text.strip():
                            fallback_section["paragraphs"].append({
                                "type": "paragraph",
                                "text": paragraph_text,
                                "coordinates": self._get_coordinates(elem)
                            })
                    elif elem.tag.endswith('list'):
                        list_data = self._extract_list(elem)
                        if list_data["items"]:
                            fallback_section["paragraphs"].append(list_data)
                
                if fallback_section["paragraphs"]:
                    sections.append(fallback_section)
        
        except Exception as e:
            self.logger.error(f"Error extracting sections: {e}")
            sections = []
        
        return sections
    
    def _determine_section_level(self, div_element) -> int:
        """Determine hierarchical level of section based on nesting."""
        level = 1
        parent = div_element.getparent()
        while parent is not None and parent.tag.endswith('div'):
            level += 1
            parent = parent.getparent()
        return min(level, 6)  # Cap at level 6
    
# Ersetzen Sie in improved_grobid.py die _extract_tables Methode (ca. Zeile 280):

    def _extract_tables(self, root) -> List[Dict[str, Any]]:
        """Extract tables with enhanced metadata."""
        tables = []
        
        # Find all figure elements
        all_figures = root.findall('.//tei:figure', self.namespaces)
        
        # Filter for tables
        table_idx = 0
        for i, fig in enumerate(all_figures):
            if fig.get('type') == 'table':
                coords = self._get_coordinates(fig)
                
                table_idx += 1
                table_data = {
                    "id": f"table_{table_idx}",
                    "caption": "",
                    "coordinates": coords,  # Kann None sein
                    "content_xml": etree.tostring(fig, encoding='unicode', pretty_print=True),
                    "table_type": fig.get('type', 'table'),
                    "label": fig.get('xml:id', ''),
                    # NEU: Seiteninformation aus dem XML extrahieren, falls keine Koordinaten
                    "estimated_page": self._estimate_page_from_context(fig) if not coords else None
                }
                
                # Extract caption from head and figDesc
                caption_parts = []
                head = fig.find('.//tei:head', self.namespaces)
                if head is not None:
                    head_text = self._get_text_content(head)
                    if head_text:
                        caption_parts.append(head_text)
                
                desc = fig.find('.//tei:figDesc', self.namespaces)
                if desc is not None:
                    desc_text = self._get_text_content(desc)
                    if desc_text:
                        caption_parts.append(desc_text)
                
                table_data["caption"] = " ".join(caption_parts).strip()
                
                # Try to extract table structure if available
                table_elem = fig.find('.//tei:table', self.namespaces)
                if table_elem is not None:
                    table_data["has_structure"] = True
                else:
                    table_data["has_structure"] = False
                
                tables.append(table_data)
        
        self.logger.debug(f"Extracted {len(tables)} tables")
        return tables
    
    # FIX 2: Neue Hilfsmethode in improved_grobid.py hinzufügen
    
    def _estimate_page_from_context(self, element) -> Optional[int]:
        """
        Versucht die Seitenzahl aus dem Kontext zu schätzen.
        Dies ist ein Fallback wenn keine Koordinaten vorhanden sind.
        """
        # Versuche pb (page break) Elemente vor diesem Element zu finden
        parent = element.getparent()
        if parent is not None:
            # Finde alle vorherigen page breaks
            preceding_pbs = parent.xpath('.//preceding::tei:pb', namespaces=self.namespaces)
            if preceding_pbs:
                last_pb = preceding_pbs[-1]
                page_n = last_pb.get('n')
                if page_n and page_n.isdigit():
                    return int(page_n)
        
        # Default: Erste Seite annehmen
        return 1
    
    # Und ähnlich für _extract_figures (ca. Zeile 330):
    
    def _extract_figures(self, root) -> List[Dict[str, Any]]:
        """Extract figures with enhanced metadata."""
        figures = []
        
        # Find all figure elements
        all_figures = root.findall('.//tei:figure', self.namespaces)
        
        # Filter out tables
        fig_idx = 0
        for i, fig in enumerate(all_figures):
            if fig.get('type') != 'table':  # Not a table
                coords = self._get_coordinates(fig)
                
                fig_idx += 1
                fig_data = {
                    "id": f"figure_{fig_idx}",
                    "caption": "",
                    "coordinates": coords,  # Kann None sein
                    "graphic_ref": None,
                    "figure_type": fig.get('type', 'figure'),
                    "label": fig.get('xml:id', ''),
                    # NEU: Seiteninformation
                    "estimated_page": self._estimate_page_from_context(fig) if not coords else None
                }
                
                # Extract caption
                caption_parts = []
                head = fig.find('.//tei:head', self.namespaces)
                if head is not None:
                    head_text = self._get_text_content(head)
                    if head_text:
                        caption_parts.append(head_text)
                
                desc = fig.find('.//tei:figDesc', self.namespaces)
                if desc is not None:
                    desc_text = self._get_text_content(desc)
                    if desc_text:
                        caption_parts.append(desc_text)
                
                fig_data["caption"] = " ".join(caption_parts).strip()
                
                # Extract graphic reference
                graphic = fig.find('.//tei:graphic', self.namespaces)
                if graphic is not None:
                    fig_data["graphic_ref"] = graphic.get('url', '')
                
                figures.append(fig_data)
        
        self.logger.debug(f"Extracted {len(figures)} figures")
        return figures
    
    def _extract_references(self, root) -> List[Dict[str, Any]]:
        """Extract bibliography references."""
        references = []
        
        # Look for bibliography section
        list_bibl = root.find('.//tei:listBibl', self.namespaces)
        if list_bibl is None:
            # Try alternative location
            divs = root.findall('.//tei:div', self.namespaces)
            for div in divs:
                if div.get('type') == 'references':
                    list_bibl = div.find('.//tei:listBibl', self.namespaces)
                    if list_bibl is not None:
                        break
        
        if list_bibl is not None:
            bibl_structs = list_bibl.findall('.//tei:biblStruct', self.namespaces)
            
            for i, bibl in enumerate(bibl_structs):
                ref_text = self._get_text_content(bibl)
                if ref_text.strip():  # Only add non-empty references
                    ref = {
                        "id": f"ref_{i+1}",
                        "text": ref_text,
                        "xml": etree.tostring(bibl, encoding='unicode', pretty_print=True),
                        "key": bibl.get('xml:id', '')
                    }
                    
                    # Try to extract structured data
                    title = bibl.find('.//tei:title', self.namespaces)
                    if title is not None:
                        ref["title"] = self._get_text_content(title)
                    
                    references.append(ref)
        
        self.logger.debug(f"Extracted {len(references)} references")
        return references
    
    def _extract_metadata(self, root) -> Dict[str, Any]:
        """Extract document metadata from TEI header."""
        metadata = {}
        
        try:
            header = root.find('.//tei:teiHeader', self.namespaces)
            if header is not None:
                # Title - look for main title first
                title_elem = None
                titles = header.findall('.//tei:title', self.namespaces)
                for t in titles:
                    if t.get('type') == 'main':
                        title_elem = t
                        break
                
                # If no main title, take first title
                if title_elem is None and titles:
                    title_elem = titles[0]
                
                if title_elem is not None:
                    metadata["title"] = self._get_text_content(title_elem)
                
                # Authors
                authors = []
                for author in header.findall('.//tei:author', self.namespaces):
                    author_text = self._get_text_content(author)
                    if author_text:
                        authors.append(author_text)
                if authors:
                    metadata["authors"] = authors
                
                # Abstract
                abstract = header.find('.//tei:abstract', self.namespaces)
                if abstract is not None:
                    metadata["abstract"] = self._get_text_content(abstract)
                
                # Keywords
                keywords = []
                for term in header.findall('.//tei:keywords//tei:term', self.namespaces):
                    keyword = self._get_text_content(term)
                    if keyword:
                        keywords.append(keyword)
                if keywords:
                    metadata["keywords"] = keywords
                    
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _parse_grobid_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse Grobid XML with comprehensive error handling."""
        if not xml_content or not xml_content.strip():
            return {"error": "Empty XML content", "raw_xml": xml_content}
        
        try:
            # Parse XML
            parser = etree.XMLParser(remove_blank_text=True, recover=True)
            root = etree.fromstring(xml_content.encode('utf-8'), parser)
            
            # Initialize structure
            structure = {
                "sections": [],
                "tables": [],
                "figures": [],
                "references": [],
                "metadata": {},
                "raw_xml": xml_content
            }
            
            # Extract metadata
            structure["metadata"] = self._extract_metadata(root)
            
            # Extract content
            body = root.find('.//tei:body', self.namespaces)
            structure["sections"] = self._extract_sections(body)
            structure["tables"] = self._extract_tables(root)
            structure["figures"] = self._extract_figures(root)
            structure["references"] = self._extract_references(root)
            
            self.logger.debug(
                f"Parsed Grobid XML: {len(structure['sections'])} sections, "
                f"{len(structure['tables'])} tables, {len(structure['figures'])} figures, "
                f"{len(structure['references'])} references"
            )
            
            return structure
            
        except etree.XMLSyntaxError as e:
            self.logger.error(f"XML parsing error: {e}")
            return {
                "error": f"XML syntax error: {str(e)}",
                "raw_xml": xml_content[:1000] + "..." if len(xml_content) > 1000 else xml_content
            }
        except Exception as e:
            self.logger.error(f"Unexpected error parsing Grobid XML: {e}")
            return {
                "error": f"Parsing error: {str(e)}",
                "raw_xml": xml_content[:1000] + "..." if len(xml_content) > 1000 else xml_content
            }
    
    @retry_with_exponential_backoff(
        max_retries=3,
        exceptions=(requests.exceptions.RequestException, requests.exceptions.Timeout)
    )
    def process_pdf(self, pdf_path: Path, consolidate_citations: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """
        Process PDF with Grobid using retry mechanism and thread-safe sessions.
        
        Args:
            pdf_path: Path to the PDF file
            consolidate_citations: Whether to consolidate citations (uses config default if None)
            
        Returns:
            Dict containing parsed structure or None on failure
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        url = f"{self.grobid_url}/api/processFulltextDocument"
        
        # Use config default if not specified
        if consolidate_citations is None:
            consolidate_citations = self.consolidate_citations
        
        try:
            with open(pdf_path, 'rb') as pdf_file:
                files = {
                    'input': (pdf_path.name, pdf_file, 'application/pdf')
                }
                data = {
                    'consolidateCitations': '1' if consolidate_citations else '0',
                    'includeRawCitations': '1',
                    'teiCoordinates': '1',
                    'segmentSentences': '1'  # Better sentence segmentation
                }
                
                # Use thread-safe session
                response = self.session_manager.post(
                    url, 
                    files=files, 
                    data=data, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                if not response.text:
                    self.logger.warning(f"Empty response from Grobid for {pdf_path.name}")
                    return None
                
                # Parse the XML response
                parsed_data = self._parse_grobid_xml(response.text)
                
                if "error" in parsed_data:
                    self.logger.error(f"Grobid parsing error for {pdf_path.name}: {parsed_data['error']}")
                    return parsed_data
                
                self.logger.info(f"Successfully processed {pdf_path.name} with Grobid")
                return parsed_data
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Grobid timeout for {pdf_path.name} (timeout: {self.timeout}s)")
            raise  # Let retry mechanism handle this
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Grobid API error for {pdf_path.name}: {e}")
            raise  # Let retry mechanism handle this
        except Exception as e:
            self.logger.error(f"Unexpected error processing {pdf_path.name} with Grobid: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        try:
            self.session_manager.close_session()
        except Exception as e:
            self.logger.warning(f"Error cleaning up Grobid processor: {e}")