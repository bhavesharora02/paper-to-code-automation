# scripts/test_gemini_extraction.py
import os
from pathlib import Path
from dotenv import load_dotenv
from scripts.ir_generation.llm_extractor import LLMExtractor

def test_with_real_paper(paper_path: str):
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set it in your .env file.")
        return
    
    # Initialize the extractor
    extractor = LLMExtractor()
    
    # Read the paper text
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            paper_text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"Processing paper: {paper_path}")
    print("=" * 80)
    
    # Extract IR
    try:
        ir = extractor.extract_ir(paper_text)
        print("Successfully extracted IR:")
        print("-" * 40)
        print(f"Title: {ir.paper_metadata.title}")
        print(f"Authors: {', '.join(ir.paper_metadata.authors) if ir.paper_metadata.authors else 'N/A'}")
        print(f"Model Type: {ir.model.type}")
        print(f"Number of Layers: {len(ir.model.layers)}")
        print(f"Training Config: {ir.training}")
        print(f"Evaluation Metrics: {ir.evaluation}")
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    # Path to the sample paper text file in the project root
    paper_path = "sample.txt"
    test_with_real_paper(paper_path)