import os
import json
from scripts.parsing.render_pages import render_pdf_to_images, extract_embedded_images
from scripts.parsing.pdf_text_extract import extract_pdf_text
from scripts.parsing.ocr_diagram_extractor import ocr_images

def run_pipeline(pdf_path: str, out_dir: str, dpi: int = 200, ocr_lang: str = "en"):
    pages_dir = os.path.join(out_dir, "pages")
    embeds_dir = os.path.join(out_dir, "embedded_images")
    ocr_pages_json = os.path.join(out_dir, "ocr_pages.json")
    ocr_embeds_json = os.path.join(out_dir, "ocr_embeds.json")
    text_json = os.path.join(out_dir, "text_sections.json")
    combined_json = os.path.join(out_dir, "combined_output.json")

    # Render & extract images
    page_imgs = render_pdf_to_images(pdf_path, pages_dir, dpi)
    embed_imgs = extract_embedded_images(pdf_path, embeds_dir)

    # OCR
    ocr_pages = ocr_images(page_imgs, ocr_lang)
    ocr_embeds = ocr_images(embed_imgs, ocr_lang) if embed_imgs else {}

    # Text extraction
    text_data = extract_pdf_text(pdf_path)

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    with open(ocr_pages_json, "w", encoding="utf-8") as f:
        json.dump(ocr_pages, f, indent=2)
    with open(ocr_embeds_json, "w", encoding="utf-8") as f:
        json.dump(ocr_embeds, f, indent=2)
    with open(text_json, "w", encoding="utf-8") as f:
        json.dump(text_data, f, indent=2)

    combined = {
        "pdf_path": pdf_path,
        "pages_dir": pages_dir,
        "embedded_images_dir": embeds_dir,
        "ocr_pages": ocr_pages,
        "ocr_embeds": ocr_embeds,
        "text": text_data
    }
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    return combined_json