"""
Wrapper script to run the Week 2 PDF parsing pipeline.
Usage: python -m scripts.run_week2
"""
import json
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.parsing.run_week2_demo import run_pipeline

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)

def main():
    # Path to the configuration file
    config_path = os.path.join(project_root, 'configs', 'week2_config.json')
    
    # Load configuration
    config = load_config(config_path)
    
    print(f"Starting PDF processing pipeline with config: {config_path}")
    print(f"Input PDF: {config.get('input_pdf')}")
    print(f"Output directory: {config.get('out_dir')}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(config['out_dir'], exist_ok=True)
        
        # Run the pipeline
        result = run_pipeline(
            pdf_path=config['input_pdf'],
            out_dir=config['out_dir'],
            dpi=config.get('render_dpi', 200),
            ocr_lang=config.get('ocr_language', 'en')
        )
        
        print(f"\nPipeline completed successfully!")
        print(f"Output saved to: {result}")
        
    except Exception as e:
        print(f"\nError during pipeline execution: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()