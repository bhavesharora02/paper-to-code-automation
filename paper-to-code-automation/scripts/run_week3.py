"""
Week 3: Intermediate Representation (IR) Generation

This script processes a PDF research paper, extracts its content, and generates
an Intermediate Representation (IR) of the model described in the paper.
"""
import os
import json
import logging
import argparse
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# Add project root to Python path
import sys
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our modules
from scripts.ir_generation.ir_generator import generate_ir_from_text
from scripts.ir_generation.model_ir import ModelIR, create_default_ir
from scripts.parsing.run_week2_demo import run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise

def extract_metadata(text_sections: dict) -> dict:
    """Extract metadata like title and authors from text sections."""
    metadata = {
        "title": "",
        "authors": [],
        "year": None,
        "url": None,
        "doi": None
    }
    
    try:
        # Try to get title from the 'title' field
        if "title" in text_sections:
            # Clean up the title (remove extra spaces, newlines, etc.)
            title = ' '.join(str(text_sections["title"]).split())
            # Sometimes the title gets concatenated with authors - split on common patterns
            for sep in [" •", " ∗", " †", " ‡"]:
                if sep in title:
                    title = title.split(sep)[0].strip()
            metadata["title"] = title
            
        # Try to extract authors from title or preamble
        if not metadata["title"] and "preamble" in text_sections:
            # If no title found, first line of preamble might be the title
            first_line = str(text_sections["preamble"]).split('\n')[0].strip()
            if first_line and len(first_line) < 200:  # Reasonable title length
                metadata["title"] = ' '.join(first_line.split())
                
        # Extract authors from preamble if available
        if "preamble" in text_sections:
            # Look for author lines (usually after title, before abstract)
            lines = str(text_sections["preamble"]).split('\n')
            author_lines = []
            found_title = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Skip the title line
                if not found_title:
                    found_title = True
                    continue
                    
                # Stop at abstract or email addresses
                if line.lower().startswith('abstract') or '@' in line:
                    break
                    
                # Add potential author line
                if line and len(line) < 100:  # Reasonable author line length
                    author_lines.append(line)
            
            # Clean up author names
            authors = []
            for line in author_lines:
                # Split on common separators
                for name in re.split(r'[\u2022\u2217\u2020\u2021\*†‡§]', line):
                    name = name.strip()
                    if (name and not name.lower().startswith('google') 
                            and not name.lower().startswith('http')):
                        authors.append(name)
            
            if authors:
                metadata["authors"] = authors
        
        # Try to extract year from abstract or text
        full_text = '\n'.join(str(v) for v in text_sections.values())
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', full_text)
        if year_match:
            metadata["year"] = int(year_match.group(1))
            
    except Exception as e:
        logger.warning(f"Error extracting metadata: {str(e)}")
    
    return metadata

def process_paper(
    pdf_path: str,
    output_dir: str,
    dpi: int = 200,
    ocr_lang: str = "en"
) -> dict:
    """
    Process a research paper PDF and generate its IR.
    
    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory to save outputs
        dpi: DPI for rendering PDF pages
        ocr_lang: Language for OCR processing
        
    Returns:
        Dictionary containing paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running PDF processing pipeline on {pdf_path}")
    
    # Run the Week 2 pipeline
    combined_json = run_pipeline(
        pdf_path=pdf_path,
        out_dir=str(output_dir),
        dpi=dpi,
        ocr_lang=ocr_lang
    )
    
    # Step 2: Load extracted text
    text_sections_path = output_dir / "text_sections.json"
    if not text_sections_path.exists():
        raise FileNotFoundError(f"Text sections not found at {text_sections_path}")
    
    with open(text_sections_path, 'r', encoding='utf-8') as f:
        try:
            text_sections = json.load(f)
            logger.info(f"Successfully loaded text sections from {text_sections_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {text_sections_path}: {e}")
            text_sections = {}
    
    # Extract metadata before processing text sections
    metadata = extract_metadata(text_sections)
    
    # Step 3: Generate IR from text
    logger.info("Generating Intermediate Representation (IR)")
    ir_output_path = output_dir / "model_ir.json"
    full_text = "\n\n".join(
        str(section) if not isinstance(section, dict) else 
        "\n".join(f"{k}: {v}" for k, v in section.items())
        for section in (text_sections.values() if isinstance(text_sections, dict) else [])
    )
    
    ir = generate_ir_from_text(full_text, str(ir_output_path))
    
    # Update IR with extracted metadata
    if hasattr(ir, 'paper_metadata'):
        for key, value in metadata.items():
            if value:  # Only update non-empty values
                setattr(ir.paper_metadata, key, value)
    
    # Save the updated IR
    with open(ir_output_path, 'w', encoding='utf-8') as f:
        json.dump(ir.dict(), f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*50)
    print("PAPER PROCESSING SUMMARY")
    print("="*50)
    print(f"Input PDF:    {pdf_path}")
    print(f"Output Dir:   {output_dir}")
    print(f"Paper Title:  {metadata.get('title', 'Not found')}")
    print(f"Authors:      {', '.join(metadata.get('authors', ['Not found']))}")
    print(f"Year:         {metadata.get('year', 'Not found')}")
    print(f"Model Name:   {getattr(ir.model, 'name', 'Unknown')}")
    print(f"Layers:       {len(getattr(ir.model, 'layers', []))}")
    
    # Safely get optimizer type
    optimizer_type = 'adam'  # Default value
    try:
        training = getattr(ir, 'training', {})
        if hasattr(training, 'optimizer'):
            optimizer = training.optimizer
            if hasattr(optimizer, 'type'):
                optimizer_type = optimizer.type
            elif hasattr(optimizer, 'name'):
                optimizer_type = optimizer.name
            elif hasattr(optimizer, '__class__'):
                optimizer_type = optimizer.__class__.__name__.replace('Optimizer', '').lower()
    except Exception as e:
        logger.warning(f"Error getting optimizer type: {e}")
    
    # Safely get learning rate
    learning_rate = getattr(getattr(ir, 'training', {}), 'learning_rate', 0.001)
    
    print(f"Optimizer:    {optimizer_type}")
    print(f"Learning Rate: {learning_rate}")
    print("="*50 + "\n")
    
    return {
        "pdf_path": str(pdf_path),
        "output_dir": str(output_dir),
        "text_sections": str(text_sections_path),
        "ir_output": str(ir_output_path),
        "ir_summary": {
            "paper_title": metadata.get("title", ""),
            "model_name": getattr(ir.model, 'name', 'Unknown'),
            "num_layers": len(getattr(ir.model, 'layers', [])),
            "optimizer": optimizer_type,
            "learning_rate": learning_rate
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Generate IR from research paper PDF")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "week3_config.json"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to input PDF (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help="DPI for rendering PDF pages (overrides config)"
    )
    parser.add_argument(
        "--ocr-lang",
        type=str,
        help="Language for OCR (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config) if os.path.exists(args.config) else {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return 1
    
    # Override config with command line arguments
    if args.pdf:
        config["input_pdf"] = args.pdf
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.dpi:
        config["dpi"] = args.dpi
    if args.ocr_lang:
        config["ocr_lang"] = args.ocr_lang
    
    # Set defaults
    input_pdf = config.get("input_pdf")
    if not input_pdf:
        logger.error("No input PDF specified. Use --pdf or set in config.")
        return 1
    
    output_dir = config.get("output_dir", "output/ir_output")
    dpi = config.get("dpi", 200)
    ocr_lang = config.get("ocr_lang", "en")
    
    try:
        result = process_paper(
            pdf_path=input_pdf,
            output_dir=output_dir,
            dpi=dpi,
            ocr_lang=ocr_lang
        )
        return 0
    except Exception as e:
        logger.error(f"Error processing paper: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())