# main_pipeline.py - Verbesserte Hauptpipeline
import sys
import os
import json
import time
import logging
import argparse
import pymupdf as fitz
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Import all modules
from extraction_config import Config, create_example_config
from resource_manager import ResourceManager
from document_pair import DocumentPair
from improved_triage import ImprovedDocumentTriageSystem
from improved_grobid import ImprovedGrobidProcessor
from improved_google_ai import ImprovedGoogleDocumentAIProcessor
from improved_image_extractor import ImprovedImageExtractor
from improved_consolidator import ImprovedDataConsolidator

logger = logging.getLogger(__name__)

class ImprovedExtractionPipeline:
    """Improved extraction pipeline with better error handling and monitoring."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize extraction pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Setup logging
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing extraction pipeline...")
        
        # Initialize resource manager
        self.resource_manager = ResourceManager(
            max_memory_mb=self.config.pipeline.max_memory_mb,
            memory_check_interval=self.config.pipeline.memory_check_interval
        )
        
        # Initialize components
        try:
            self.triage_system = ImprovedDocumentTriageSystem(self.config, self.resource_manager)
            self.grobid_processor = ImprovedGrobidProcessor(self.config)
            self.image_extractor = ImprovedImageExtractor(self.config, self.resource_manager)
            self.data_consolidator = ImprovedDataConsolidator(self.config, self.resource_manager)
            
            # Initialize Google processor only if configured
            if self._is_google_configured():
                self.google_processor = ImprovedGoogleDocumentAIProcessor(self.config)
            else:
                self.google_processor = None
                self.logger.warning("Google Document AI not configured - table/OCR extraction will be skipped")
            
            self.max_workers = self.config.pipeline.max_workers
            
            self.logger.info("Extraction pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.logging.level.upper())
        log_file = self.config.logging.file
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers
        root_logger.handlers = []
        
        # File handler
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    def _is_google_configured(self) -> bool:
        """Check if Google Document AI is enabled and properly configured."""
        return (
            self.config.google.enabled and
            self.config.google.project_id != "your-google-project-id" and
            not self.config.google.project_id.startswith("<") and
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is not None
        )
    
    def _process_single_document(self, pair: DocumentPair) -> Dict[str, Any]:
        """Process a single document through stages 2-4 with comprehensive error handling."""
        pdf_name = pair.pdf_path.name
        result = {
            "success": False,
            "document_id": pair.pdf_path.stem,
            "pdf_path": str(pair.pdf_path),
            "error": None,
            "processing_time": 0,
            "stages_completed": [],
            "warnings": []
        }
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing {pdf_name} (scan: {pair.is_scan}, {pair.file_size_mb:.1f}MB)")
            
            # Stage 2: Grobid processing
            self.logger.debug(f"Stage 2: Grobid processing for {pdf_name}")
            grobid_data = None
            
            try:
                grobid_data = self.grobid_processor.process_pdf(pair.pdf_path)
                if grobid_data:
                    result["stages_completed"].append("grobid")
                else:
                    result["warnings"].append("Grobid returned no data")
            except Exception as e:
                self.logger.error(f"Grobid processing failed for {pdf_name}: {e}")
                result["warnings"].append(f"Grobid error: {str(e)}")
            
            # Initialize extraction results
            table_extractions = {}
            figure_extractions = {}
            ocr_data = None
            
            # Stage 3: Precision extraction
            if self.google_processor:
                if pair.is_scan:
                    # OCR processing for scanned documents
                    self.logger.debug(f"Stage 3a: OCR processing for scanned document {pdf_name}")
                    try:
                        ocr_data = self.google_processor.process_scanned_pdf(pair.pdf_path)
                        if ocr_data:
                            result["stages_completed"].append("ocr")
                        else:
                            result["warnings"].append("OCR returned no data")
                    except Exception as e:
                        self.logger.warning(f"OCR failed for {pdf_name}: {e}")
                        result["warnings"].append(f"OCR error: {str(e)}")
                else:
                    # FIX 4: Verbesserte Table extraction für native PDFs
                    if grobid_data and "tables" in grobid_data and grobid_data["tables"]:
                        self.logger.debug(f"Stage 3a: Table extraction for {pdf_name}")
                        
                        # NEU: Alle Seiten sammeln, auch ohne Koordinaten
                        pages_with_tables = set()
                        estimated_pages_with_tables = set()
                        
                        for t in grobid_data["tables"]:
                            if t.get("coordinates") and "page" in t["coordinates"]:
                                pages_with_tables.add(t["coordinates"]["page"])
                            elif t.get("estimated_page"):
                                estimated_pages_with_tables.add(t["estimated_page"])
                                self.logger.info(f"Table without coordinates, trying page {t['estimated_page']}")
                        
                        # Wenn keine Seiten mit Koordinaten gefunden wurden, alle geschätzten Seiten verwenden
                        if not pages_with_tables and estimated_pages_with_tables:
                            pages_with_tables = estimated_pages_with_tables
                            self.logger.warning(f"No table coordinates from Grobid, trying estimated pages: {pages_with_tables}")
                        
                        # Falls immer noch keine Seiten, versuche alle Seiten des PDFs
                        if not pages_with_tables:
                            try:
                                import pymupdf as fitz  # PyMuPDF
                                with fitz.open(pair.pdf_path) as doc:
                                    total_pages = doc.page_count
                                    # Beschränke auf max 10 Seiten für Performance
                                    pages_with_tables = set(range(1, min(total_pages + 1, 11)))
                                    self.logger.warning(f"No table locations found, scanning first {len(pages_with_tables)} pages")
                            except:
                                pages_with_tables = {1, 2, 3}  # Fallback: erste 3 Seiten
                        
                        for page_num in sorted(pages_with_tables):
                            try:
                                tables = self.google_processor.extract_table_from_page(pair.pdf_path, page_num)
                                if tables:
                                    table_extractions[page_num] = tables
                            except Exception as e:
                                self.logger.warning(f"Table extraction failed for {pdf_name} page {page_num}: {e}")
                                result["warnings"].append(f"Table extraction error on page {page_num}: {str(e)}")
                        
                        if table_extractions:
                            result["stages_completed"].append("table_extraction")
            
            # FIX 5: Verbesserte Figure extraction (always attempt if we have Grobid data)
            if grobid_data and "figures" in grobid_data and grobid_data["figures"]:
                self.logger.debug(f"Stage 3b: Figure extraction for {pdf_name}")
                
                # NEU: Versuche auch Figures ohne Koordinaten zu extrahieren
                figures_extracted = 0
                figures_skipped = 0
                
                for fig in grobid_data["figures"]:
                    fig_id = fig.get("id")
                    bbox = fig.get("coordinates")
                    
                    if fig_id and bbox:
                        # Normal extraction with coordinates
                        try:
                            extracted_fig = self.image_extractor.extract_figure(pair.pdf_path, bbox, fig_id)
                            if extracted_fig:
                                figure_extractions[fig_id] = extracted_fig
                                figures_extracted += 1
                        except Exception as e:
                            self.logger.warning(f"Figure extraction failed for {pdf_name} {fig_id}: {e}")
                            result["warnings"].append(f"Figure extraction error for {fig_id}: {str(e)}")
                    elif fig_id and fig.get("estimated_page"):
                        # NEU: Fallback für Figures ohne Koordinaten
                        self.logger.warning(f"Figure {fig_id} has no coordinates, trying page-based extraction")
                        try:
                            page_num = fig["estimated_page"]
                            # Extrahiere alle Bilder von der geschätzten Seite
                            page_images = self.image_extractor.extract_all_images_from_page(pair.pdf_path, page_num)
                            if page_images:
                                # Nehme das größte Bild (wahrscheinlich die Hauptfigur)
                                largest_img = max(page_images, key=lambda x: x.get('width', 0) * x.get('height', 0))
                                figure_extractions[fig_id] = largest_img
                                figures_extracted += 1
                                self.logger.info(f"Extracted figure {fig_id} using page-based fallback")
                        except Exception as e:
                            self.logger.warning(f"Page-based extraction failed for {fig_id}: {e}")
                            figures_skipped += 1
                    else:
                        figures_skipped += 1
                        self.logger.warning(f"Figure {fig_id} has no usable location information")
                
                if figures_extracted > 0:
                    result["stages_completed"].append("figure_extraction")
                
                if figures_skipped > 0:
                    result["warnings"].append(f"Skipped {figures_skipped} figures without coordinates")
                
                self.logger.info(f"Extracted {figures_extracted} figures, skipped {figures_skipped}")
            
            # Stage 4: Data consolidation
            self.logger.debug(f"Stage 4: Data consolidation for {pdf_name}")
            
            final_document = self.data_consolidator.consolidate_document(
                pair, grobid_data, table_extractions, figure_extractions, ocr_data
            )
            
            result["stages_completed"].append("consolidation")
            
            # Save document
            save_success = self.data_consolidator.save_consolidated_document(final_document)
            if save_success:
                result["stages_completed"].append("save")
                result["success"] = True
                result["statistics"] = final_document.get("statistics", {})
            else:
                result["error"] = "Failed to save consolidated document"
            
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            self.logger.exception(f"Unexpected error processing {pdf_name}")
        finally:
            result["processing_time"] = time.time() - start_time
            
            # Log processing summary
            self.logger.info(
                f"Completed {pdf_name}: success={result['success']}, "
                f"stages={len(result['stages_completed'])}, "
                f"time={result['processing_time']:.1f}s"
            )
        
        return result
    
    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete extraction pipeline with comprehensive reporting.
        
        Args:
            limit: Optional limit on number of documents to process
            
        Returns:
            Pipeline statistics
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING EXTRACTION PIPELINE")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Log system information
        system_info = self.resource_manager.get_system_info()
        if system_info:
            self.logger.info(f"System info: {system_info}")
        
        # Stage 1: Document triage
        self.logger.info("Stage 1: Document triage and validation")
        valid_pairs, invalid_pairs = self.triage_system.process_batch(self.max_workers)
        
        # Apply limit if specified
        if limit and limit < len(valid_pairs):
            self.logger.info(f"Limiting processing to {limit} documents")
            valid_pairs = valid_pairs[:limit]
        
        pipeline_stats = {
            "start_time": start_time.isoformat(),
            "configuration": {
                "max_workers": self.max_workers,
                "max_memory_mb": self.config.pipeline.max_memory_mb,
                "google_ai_enabled": self.google_processor is not None
            },
            "total_documents": len(valid_pairs) + len(invalid_pairs),
            "valid_documents": len(valid_pairs),
            "invalid_documents": len(invalid_pairs),
            "processed_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "processing_results": [],
            "memory_stats": [],
            "warnings_summary": {}
        }
        
        if not valid_pairs:
            self.logger.info("No valid documents found for processing")
            pipeline_stats["end_time"] = datetime.now().isoformat()
            pipeline_stats["total_processing_time"] = 0
            return pipeline_stats
        
        self.logger.info(f"Processing {len(valid_pairs)} valid documents...")
        
        # Process documents
        processed_documents = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='ProcessorWorker') as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(self._process_single_document, pair): pair
                for pair in valid_pairs
            }
            
            # Process results with progress bar
            with tqdm(total=len(valid_pairs), desc="Stages 2-4: Document Processing") as pbar:
                for future in as_completed(future_to_pair):
                    try:
                        result = future.result()
                        pipeline_stats["processing_results"].append(result)
                        pipeline_stats["processed_documents"] += 1
                        
                        if result["success"]:
                            pipeline_stats["successful_documents"] += 1
                            processed_documents.append(result)
                        else:
                            pipeline_stats["failed_documents"] += 1
                            self.logger.error(
                                f"Failed to process {result['document_id']}: "
                                f"{result.get('error', 'Unknown error')}"
                            )
                        
                        # Track warnings
                        for warning in result.get("warnings", []):
                            warning_type = warning.split(":")[0]
                            pipeline_stats["warnings_summary"][warning_type] = \
                                pipeline_stats["warnings_summary"].get(warning_type, 0) + 1
                    
                    except Exception as e:
                        pair = future_to_pair[future]
                        self.logger.error(f"Task execution failed for {pair.pdf_path.name}: {e}")
                        pipeline_stats["failed_documents"] += 1
                        pipeline_stats["processing_results"].append({
                            "success": False,
                            "document_id": pair.pdf_path.stem,
                            "pdf_path": str(pair.pdf_path),
                            "error": f"Task execution error: {str(e)}",
                            "processing_time": 0,
                            "stages_completed": []
                        })
                    finally:
                        pbar.update(1)
                        
                        # Periodic memory check
                        if pipeline_stats["processed_documents"] % 10 == 0:
                            memory_stats = self.resource_manager.get_memory_stats()
                            if memory_stats:
                                pipeline_stats["memory_stats"].append({
                                    "documents_processed": pipeline_stats["processed_documents"],
                                    "stats": memory_stats
                                })
        
        # Final statistics
        end_time = datetime.now()
        pipeline_stats["end_time"] = end_time.isoformat()
        pipeline_stats["total_processing_time"] = (end_time - start_time).total_seconds()
        
        # Final memory stats
        final_memory_stats = self.resource_manager.get_memory_stats()
        if final_memory_stats:
            pipeline_stats["final_memory_stats"] = final_memory_stats
        
        # Calculate success rate
        if pipeline_stats["processed_documents"] > 0:
            pipeline_stats["success_rate"] = (
                pipeline_stats["successful_documents"] / pipeline_stats["processed_documents"]
            )
        else:
            pipeline_stats["success_rate"] = 0.0
        
        # Save pipeline statistics
        self._save_pipeline_stats(pipeline_stats)
        
        # Generate and save consolidated report if documents were processed
        if processed_documents:
            self._generate_consolidated_report(processed_documents)
        
        # Log summary
        self._log_summary(pipeline_stats)
        
        # Cleanup
        self.resource_manager.force_cleanup()
        
        return pipeline_stats
    
    def _save_pipeline_stats(self, stats: Dict[str, Any]):
        """Save comprehensive pipeline statistics."""
        try:
            stats_path = Path(self.config.paths.output_dir) / "pipeline_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Pipeline statistics saved to {stats_path}")
        except Exception as e:
            self.logger.error(f"Could not save pipeline statistics: {e}")
    
    def _generate_consolidated_report(self, processed_documents: List[Dict[str, Any]]):
        """Generate consolidated report for all processed documents."""
        try:
            # Load all extracted documents
            documents = []
            for result in processed_documents:
                if result["success"]:
                    doc_path = Path(self.config.paths.output_dir) / f"{result['document_id']}_extracted.json"
                    if doc_path.exists():
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            documents.append(json.load(f))
            
            if documents:
                summary_report = self.data_consolidator.generate_summary_report(documents)
                report_path = Path(self.config.paths.output_dir) / "extraction_summary_report.json"
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_report, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Summary report saved to {report_path}")
                
        except Exception as e:
            self.logger.error(f"Could not generate consolidated report: {e}")
    
    def _log_summary(self, stats: Dict[str, Any]):
        """Log pipeline execution summary."""
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total documents: {stats['total_documents']}")
        self.logger.info(f"Valid documents: {stats['valid_documents']}")
        self.logger.info(f"Successfully processed: {stats['successful_documents']}")
        self.logger.info(f"Failed: {stats['failed_documents']}")
        self.logger.info(f"Invalid (triage): {stats['invalid_documents']}")
        self.logger.info(f"Success rate: {stats['success_rate']:.1%}")
        self.logger.info(f"Total processing time: {stats['total_processing_time']:.1f}s")
        
        if stats["warnings_summary"]:
            self.logger.info("Warnings summary:")
            for warning_type, count in stats["warnings_summary"].items():
                self.logger.info(f"  - {warning_type}: {count}")
        
        if "final_memory_stats" in stats:
            self.logger.info(
                f"Memory usage: {stats['final_memory_stats']['current_rss_mb']:.1f}MB "
                f"(peak: {stats['final_memory_stats']['peak_memory_mb']:.1f}MB)"
            )

def main():
    """Main entry point."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Scientific PDF Extraction Pipeline v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create example configuration
  python main_pipeline.py --create-config
  
  # Run pipeline with default config
  python main_pipeline.py
  
  # Run pipeline with custom config
  python main_pipeline.py --config my_config.yaml
  
  # Process only first 10 documents
  python main_pipeline.py --limit 10
  
  # Run with debug logging
  python main_pipeline.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--create-config", 
        action="store_true", 
        help="Create example configuration file and exit"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of documents to process"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="PDF Extraction Pipeline v2.0"
    )
    
    args = parser.parse_args()
    
    # Create example config if requested
    if args.create_config:
        create_example_config("config.yaml")
        sys.exit(0)
    
    # Set logging level from command line
    if args.log_level:
        os.environ['LOG_LEVEL'] = args.log_level
    
    try:
        # Run pipeline
        pipeline = ImprovedExtractionPipeline(args.config)
        results = pipeline.run(limit=args.limit)
        
        # Exit with appropriate code
        if results["failed_documents"] == 0:
            sys.exit(0)  # Success
        elif results["successful_documents"] > 0:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # Complete failure
            
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()