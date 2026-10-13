# config.py - Externe Konfigurationsverwaltung
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class PathConfig:
    """Configuration for file paths."""
    pdf_dir: Path
    metadata_dir: Path
    output_dir: Path
    image_dir: Optional[Path] = None
    progress_db: Optional[Path] = None
    
    def __post_init__(self):
        # Convert strings to Path objects
        self.pdf_dir = Path(self.pdf_dir)
        self.metadata_dir = Path(self.metadata_dir)
        self.output_dir = Path(self.output_dir)
        
        if self.image_dir is None:
            self.image_dir = self.output_dir / "images"
        else:
            self.image_dir = Path(self.image_dir)
            
        if self.progress_db is None:
            self.progress_db = self.output_dir / "pipeline_progress.db"
        else:
            self.progress_db = Path(self.progress_db)
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class GrobidConfig:
    """Configuration for Grobid service."""
    url: str
    timeout: int = 300
    max_retries: int = 3
    consolidate_citations: bool = True
    
@dataclass
class GoogleConfig:
    """Configuration for Google Document AI."""
    project_id: str
    location: str
    table_processor_id: str
    ocr_processor_id: str
    timeout: int = 300
    max_retries: int = 3
    enabled: bool = False  # Document AI ist kostenpflichtig, nur bewusst einschalten

@dataclass
class PipelineConfig:
    """Configuration for pipeline processing."""
    max_workers: int = 4
    max_memory_mb: int = 2048
    batch_size: int = 100
    enable_retry: bool = True
    memory_check_interval: int = 10
    
@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file: str = "pipeline.log"

class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('PIPELINE_CONFIG', 'config.yaml')
        self._raw_config = self._load_config()
        self._validate_config()
        
        # Parse structured config
        self.paths = PathConfig(**self._raw_config['paths'])
        self.grobid = GrobidConfig(**self._raw_config['grobid'])
        self.google = GoogleConfig(**self._raw_config['google'])
        self.pipeline = PipelineConfig(**self._raw_config.get('pipeline', {}))
        self.logging = LoggingConfig(**self._raw_config.get('logging', {}))
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            # Create default config if it doesn't exist
            default_config = self._get_default_config()
            self._save_config(default_config, config_path)
            return default_config
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                else:
                    return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse config file {config_path}: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "paths": {
                "pdf_dir": "./pdfs",
                "metadata_dir": "./metadata",
                "output_dir": "./output",
                "image_dir": "./output/images",
                "progress_db": "./output/pipeline_progress.db"
            },
            "grobid": {
                "url": "http://localhost:8070",
                "timeout": 300,
                "max_retries": 3,
                "consolidate_citations": True
            },
            "google": {
                "enabled": False,
                "project_id": "your-google-project-id",
                "location": "eu",
                "table_processor_id": "your-table-processor-id",
                "ocr_processor_id": "your-ocr-processor-id",
                "timeout": 300,
                "max_retries": 3
            },
            "pipeline": {
                "max_workers": 4,
                "max_memory_mb": 2048,
                "batch_size": 100,
                "enable_retry": True,
                "memory_check_interval": 10
            },
            "logging": {
                "level": "INFO",
                "file": "pipeline.log"
            }
        }
    
    def _save_config(self, config: Dict[str, Any], path: Path):
        """Save configuration to file."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    json.dump(config, f, indent=2)
            print(f"Created default configuration at: {path}")
        except Exception as e:
            raise ValueError(f"Failed to save configuration: {e}")
    
    def _validate_config(self):
        """Validate required configuration sections."""
        required_sections = ['paths', 'grobid', 'google']
        missing_sections = [s for s in required_sections if s not in self._raw_config]
        
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")
        
        # Validate paths
        paths = self._raw_config['paths']
        for path_name in ['pdf_dir', 'metadata_dir']:
            if path_name not in paths:
                raise ValueError(f"Missing required path: {path_name}")
            
            path = Path(paths[path_name])
            if not path.exists():
                print(f"Warning: Path does not exist: {path} (will be created if needed)")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self._raw_config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

def create_example_config(config_path: str = "config.yaml"):
    """Create an example configuration file."""
    config = Config(config_path)
    print(f"Example configuration created at: {config_path}")
    print("Please edit the configuration file with your actual values.")