import os
import sys
from pathlib import Path

import cv2
from PIL import Image
import pytesseract

def preprocess(img_path: Path) -> Image.Image:
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"ERROR: cannot read image '{img_path}'")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(thresh, 3)
    # Convert back to PIL.Image for pytesseract compatibility
    return Image.fromarray(denoised)

def extract_text(img: Image.Image) -> str:
    return pytesseract.image_to_string(img, lang="eng")

def save_text(text: str, out_path: Path) -> None:
    """
    Save the extracted text to a .txt file (UTF‑8 encoded).
    """
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text.strip())

def main():
    root = Path(__file__).parent.parent
    sample = root / "samples" / "sample_img.png"
    output = root / "outputs" / "extracted_text.txt"

    print(f"[INFO] Preprocessing '{sample}' …", flush=True)
    img_pre = preprocess(sample)
    print(f"[INFO] Extracting text …", flush=True)
    txt = extract_text(img_pre)

    print(f"[INFO] Saving text to '{output}'", flush=True)
    save_text(txt, output)
    print("✅ Done!")

if __name__ == "__main__":
    main()
