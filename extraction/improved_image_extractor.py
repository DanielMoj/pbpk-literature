# improved_image_extractor.py - Verbesserte Bildextraktion
import pymupdf as fitz
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import logging
import io
from PIL import Image
import numpy as np

from resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class ImprovedImageExtractor:
    """Improved image extractor with better quality detection and error handling."""
    
    def __init__(self, config, resource_manager: ResourceManager):
        """
        Initialize image extractor.
        
        Args:
            config: Configuration object
            resource_manager: Resource manager for memory management
        """
        self.output_dir = Path(config.paths.image_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.min_width = 50  # Minimum width in pixels
        self.min_height = 50  # Minimum height in pixels
        self.min_quality_score = 0.3
        self.max_image_size_mb = 10  # Maximum size for extracted images
    
    def _estimate_dpi(self, pix: fitz.Pixmap, bbox: fitz.Rect) -> int:
        """Estimate DPI based on pixels and physical size."""
        try:
            # Assume 72 DPI is standard PDF unit
            width_inch = bbox.width / 72.0
            height_inch = bbox.height / 72.0
            
            if width_inch > 0 and height_inch > 0:
                # Calculate DPI based on both dimensions and take average
                dpi_width = int(pix.width / width_inch)
                dpi_height = int(pix.height / height_inch)
                estimated_dpi = (dpi_width + dpi_height) // 2
                return max(72, min(estimated_dpi, 600))  # Clamp between 72-600 DPI
        except Exception as e:
            self.logger.debug(f"Error estimating DPI: {e}")
        return 150  # Default fallback
    
    def _calculate_overlap_score(self, bbox1: fitz.Rect, bbox2: fitz.Rect) -> float:
        """Calculate intersection over union (IoU) score for two bounding boxes."""
        try:
            # Calculate intersection
            intersect = bbox1.intersect(bbox2)
            if intersect.is_empty:
                return 0.0
            
            intersect_area = intersect.width * intersect.height
            bbox1_area = bbox1.width * bbox1.height
            bbox2_area = bbox2.width * bbox2.height
            
            if bbox1_area <= 0 or bbox2_area <= 0:
                return 0.0
            
            # Calculate union area
            union_area = bbox1_area + bbox2_area - intersect_area
            
            # Calculate IoU
            if union_area > 0:
                return intersect_area / union_area
            else:
                return 0.0
                
        except Exception as e:
            self.logger.debug(f"Error calculating overlap score: {e}")
            return 0.0
    
    def _find_best_matching_image(self, page: fitz.Page, target_bbox: Dict[str, float]) -> Optional[Tuple]:
        """Find the best matching embedded image for the given bounding box."""
        target_rect = fitz.Rect(target_bbox["x1"], target_bbox["y1"], target_bbox["x2"], target_bbox["y2"])
        images = page.get_images(full=True)
        
        if not images:
            return None
        
        best_match = None
        max_score = 0.0
        
        for img in images:
            try:
                # Get image bounding box
                img_bbox = page.get_image_bbox(img)
                
                # Calculate overlap score
                iou_score = self._calculate_overlap_score(img_bbox, target_rect)
                
                # Also consider how much of the target is covered
                intersect = img_bbox.intersect(target_rect)
                if not intersect.is_empty:
                    target_coverage = (intersect.width * intersect.height) / (target_rect.width * target_rect.height)
                else:
                    target_coverage = 0.0
                
                # Combined score
                score = (iou_score * 0.7) + (target_coverage * 0.3)
                
                if score > max_score and score > 0.3:  # Minimum threshold
                    max_score = score
                    best_match = (img, img_bbox, score)
                    
            except Exception as e:
                self.logger.debug(f"Error checking image overlap: {e}")
                continue
        
        return best_match
    
    def _convert_pixmap_to_pil(self, pix: fitz.Pixmap) -> Optional[Image.Image]:
        """Convert PyMuPDF Pixmap to PIL Image."""
        try:
            # Convert to RGB if necessary
            if pix.n - pix.alpha >= 4:  # CMYK or more complex
                pix = fitz.Pixmap(fitz.csRGB, pix)
            
            # Get image data
            img_data = pix.tobytes("png")
            
            # Create PIL Image
            img = Image.open(io.BytesIO(img_data))
            return img
            
        except Exception as e:
            self.logger.error(f"Error converting pixmap to PIL: {e}")
            return None
    
    def _assess_image_quality(self, pix: fitz.Pixmap, match_score: float = 1.0) -> float:
        """Assess image quality based on resolution and content analysis."""
        quality = 0.5  # Base quality
        
        try:
            # Resolution factor
            pixel_count = pix.width * pix.height
            if pixel_count > 1000000:  # > 1MP
                quality += 0.3
            elif pixel_count > 500000:  # > 0.5MP
                quality += 0.2
            elif pixel_count > 250000:  # > 0.25MP
                quality += 0.1
            elif pixel_count < 10000:  # < 0.01MP (very small)
                quality -= 0.2
            
            # Size checks
            if pix.width < self.min_width or pix.height < self.min_height:
                quality -= 0.3
            
            # Aspect ratio (reasonable ratios get bonus)
            if pix.width > 0 and pix.height > 0:
                aspect_ratio = max(pix.width, pix.height) / min(pix.width, pix.height)
                if aspect_ratio < 3:  # Not too extreme
                    quality += 0.1
                elif aspect_ratio > 10:  # Very extreme
                    quality -= 0.1
            
            # Match score factor
            quality += match_score * 0.2
            
            # Analyze image content using PIL
            pil_img = self._convert_pixmap_to_pil(pix)
            if pil_img:
                # Convert to grayscale for analysis
                gray_img = pil_img.convert('L')
                img_array = np.array(gray_img)
                
                # Check for blank/uniform images
                std_dev = np.std(img_array)
                if std_dev < 5:  # Very low variation
                    quality -= 0.3
                elif std_dev > 50:  # Good variation
                    quality += 0.1
                
        except Exception as e:
            self.logger.debug(f"Error in quality assessment: {e}")
        
        return max(0.0, min(1.0, quality))

    def _extract_raw_image(self, page: fitz.Page, bbox: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Extract embedded image with quality assessment."""
        best_match = self._find_best_matching_image(page, bbox)
        
        if not best_match:
            return None
        
        img_info, img_bbox, match_score = best_match
        
        try:
            xref = img_info[0]
            pix = fitz.Pixmap(page.parent, xref)
            
            # Check if we need to clean up alpha channel
            if pix.alpha:
                pix = fitz.Pixmap(pix, 0)  # Remove alpha
            
            # Convert CMYK to RGB if necessary
            if pix.n - pix.alpha >= 4:  # CMYK or more complex
                converted_pix = fitz.Pixmap(fitz.csRGB, pix)
                pix = converted_pix
            
            # Quality assessment
            quality_score = self._assess_image_quality(pix, match_score)
            
            if quality_score < self.min_quality_score:
                self.logger.debug(f"Image quality too low: {quality_score:.2f}")
                return None
            
            # Get image data
            img_data = pix.tobytes("png")
            
            # Check size
            size_mb = len(img_data) / (1024 * 1024)
            if size_mb > self.max_image_size_mb:
                self.logger.warning(f"Image too large: {size_mb:.1f}MB")
                return None
            
            result = {
                "type": "raw",
                "format": "png",
                "data": img_data,
                "width": pix.width,
                "height": pix.height,
                "dpi": self._estimate_dpi(pix, img_bbox),
                "quality_score": quality_score,
                "match_score": match_score,
                "size_mb": size_mb
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting raw image: {e}")
            return None
        finally:
            # Clean up pixmaps
            if 'pix' in locals():
                pix = None
    
    def _render_area(self, page: fitz.Page, bbox: Dict[str, float], figure_id: str, doc_name: str) -> Dict[str, Any]:
        """Render area in high resolution with adaptive DPI."""
        rect = fitz.Rect(bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
        
        # Validate rectangle
        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            raise ValueError(f"Invalid bounding box for {figure_id}")
        
        # Adaptive DPI based on area size
        area = rect.width * rect.height
        if area > 50000:  # Large area
            matrix = fitz.Matrix(2, 2)  # ~144 DPI
            dpi = 144
        elif area > 10000:  # Medium area
            matrix = fitz.Matrix(3, 3)  # ~216 DPI
            dpi = 216
        else:  # Small area
            matrix = fitz.Matrix(4, 4)  # ~288 DPI
            dpi = 288
        
        try:
            # Render the area
            pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
            
            # Quality check
            if pix.width < self.min_width or pix.height < self.min_height:
                self.logger.warning(f"Rendered image too small: {pix.width}x{pix.height}")
            
            # Save to file
            doc_image_dir = self.output_dir / doc_name
            doc_image_dir.mkdir(exist_ok=True)
            output_path = doc_image_dir / f"{figure_id}.png"
            
            pix.save(output_path)
            
            # Get file size
            size_mb = output_path.stat().st_size / (1024 * 1024)
            
            result = {
                "type": "rendered",
                "format": "png",
                "path": str(output_path),
                "width": pix.width,
                "height": pix.height,
                "dpi": dpi,
                "quality_score": 0.8,  # Rendered images generally good quality
                "size_mb": size_mb
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error rendering area: {e}")
            raise
        finally:
            # Clean up
            if 'pix' in locals():
                pix = None
    
    def extract_figure(self, pdf_path: Path, bbox: Dict[str, float], figure_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract figure in highest available quality.
        
        Args:
            pdf_path: Path to PDF file
            bbox: Bounding box coordinates with page number
            figure_id: Unique figure identifier
            
        Returns:
            Dict with image data or None on failure
        """
        pdf_path = Path(pdf_path)
        
        # Validate inputs
        if not bbox or not all(k in bbox for k in ["page", "x1", "y1", "x2", "y2"]):
            self.logger.warning(f"Invalid bounding box for {figure_id} in {pdf_path.name}")
            return None
        
        # Validate coordinates
        if bbox["x2"] <= bbox["x1"] or bbox["y2"] <= bbox["y1"]:
            self.logger.warning(f"Invalid coordinate order for {figure_id}")
            return None
        
        try:
            with self.resource_manager.managed_document(pdf_path) as doc:
                page_idx = bbox["page"] - 1  # Convert to 0-based
                
                if page_idx < 0 or page_idx >= doc.page_count:
                    self.logger.error(f"Page {bbox['page']} out of range for {pdf_path.name}")
                    return None
                
                page = doc[page_idx]
                doc_name = pdf_path.stem
                
                # Try to extract raw image first (higher quality)
                self.logger.debug(f"Attempting raw extraction for {figure_id}")
                raw_image = self._extract_raw_image(page, bbox)
                
                if raw_image and raw_image.get("quality_score", 0) > 0.6:
                    self.logger.info(
                        f"Extracted raw image for {figure_id} "
                        f"(quality: {raw_image['quality_score']:.2f}, "
                        f"size: {raw_image.get('size_mb', 0):.1f}MB)"
                    )
                    return raw_image
                
                # Fallback to rendering
                self.logger.debug(f"Rendering figure {figure_id} (raw extraction failed or low quality)")
                rendered_image = self._render_area(page, bbox, figure_id, doc_name)
                
                self.logger.info(
                    f"Rendered figure {figure_id} "
                    f"({rendered_image['width']}x{rendered_image['height']} @ {rendered_image['dpi']}dpi)"
                )
                return rendered_image
                
        except Exception as e:
            self.logger.error(f"Error extracting figure {figure_id} from {pdf_path.name}: {e}")
            return None
    
    def extract_all_images_from_page(self, pdf_path: Path, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract all images from a specific page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-based)
            
        Returns:
            List of extracted images with metadata
        """
        pdf_path = Path(pdf_path)
        extracted_images = []
        
        try:
            with self.resource_manager.managed_document(pdf_path) as doc:
                page_idx = page_num - 1
                
                if page_idx < 0 or page_idx >= doc.page_count:
                    self.logger.error(f"Page {page_num} out of range")
                    return extracted_images
                
                page = doc[page_idx]
                images = page.get_images(full=True)
                
                for i, img_info in enumerate(images):
                    try:
                        xref = img_info[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert if necessary
                        if pix.n - pix.alpha >= 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Get bounding box
                        bbox = page.get_image_bbox(img_info)
                        
                        # Quality assessment
                        quality_score = self._assess_image_quality(pix)
                        
                        if quality_score >= self.min_quality_score:
                            img_data = {
                                "image_index": i + 1,
                                "page": page_num,
                                "xref": xref,
                                "width": pix.width,
                                "height": pix.height,
                                "quality_score": quality_score,
                                "bbox": {
                                    "x1": bbox.x0,
                                    "y1": bbox.y0,
                                    "x2": bbox.x1,
                                    "y2": bbox.y1
                                }
                            }
                            extracted_images.append(img_data)
                        
                    except Exception as e:
                        self.logger.error(f"Error extracting image {i+1} from page {page_num}: {e}")
                        continue
                    finally:
                        if 'pix' in locals():
                            pix = None
                
                self.logger.info(f"Extracted {len(extracted_images)} images from page {page_num}")
                
        except Exception as e:
            self.logger.error(f"Error processing page {page_num}: {e}")
        
        return extracted_images