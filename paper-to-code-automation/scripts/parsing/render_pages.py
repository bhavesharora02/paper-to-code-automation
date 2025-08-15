import os
import fitz  # PyMuPDF
from tqdm import tqdm

def render_pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 200):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_images = []
    for i, page in enumerate(tqdm(doc, desc="Rendering pages")):
        mat = fitz.Matrix(dpi/72, dpi/72)  # dpi scaling
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = os.path.join(out_dir, f"page_{i+1:03d}.png")
        pix.save(img_path)
        page_images.append(img_path)
    doc.close()
    return page_images

def extract_embedded_images(pdf_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    extracted = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:  # GRAY or RGB
                img_path = os.path.join(out_dir, f"page{page_index+1:03d}_img{img_index+1:02d}.png")
                pix.save(img_path)
                extracted.append(img_path)
            else:  # CMYK
                pix = fitz.Pixmap(fitz.csRGB, pix)
                img_path = os.path.join(out_dir, f"page{page_index+1:03d}_img{img_index+1:02d}.png")
                pix.save(img_path)
                extracted.append(img_path)
    doc.close()
    return extracted