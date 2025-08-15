import os
from tqdm import tqdm
from paddleocr import PaddleOCR

def ocr_images(images, lang="en"):
    ocr = PaddleOCR(lang=lang)
    results = {}
    for img in tqdm(images, desc="OCR on images"):
        try:
            res = ocr.ocr(img, cls=True)
        except Exception:
            res = [["", 0.0]]
        lines = []
        if res:
            for page in res:
                if isinstance(page, list):
                    for item in page:
                        if isinstance(item, list) and len(item) >= 2:
                            text = item[1][0]
                            conf = item[1][1]
                            lines.append({"text": text, "conf": float(conf)})
        results[os.path.basename(img)] = lines
    return results