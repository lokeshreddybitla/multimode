"""
OCR utilities for scanned documents and handwritten text.
Provides enhanced OCR pipeline with preprocessing.
"""

import io
from typing import Optional, Dict, Any


def ocr_image(image_bytes: bytes, language: str = "eng") -> Dict[str, Any]:
    """
    Perform full OCR on an image with quality assessment.
    Returns dict with text, confidence, and metadata.
    """
    result = {
        "text": "",
        "confidence": 0.0,
        "word_count": 0,
        "method": "none",
        "success": False,
    }

    # Try pytesseract with preprocessing
    try:
        text, confidence = _tesseract_ocr(image_bytes, language)
        if text and len(text.strip()) > 10:
            result["text"] = text
            result["confidence"] = confidence
            result["word_count"] = len(text.split())
            result["method"] = "tesseract"
            result["success"] = True
            return result
    except Exception:
        pass

    # Try EasyOCR as fallback
    try:
        text = _easyocr_fallback(image_bytes)
        if text:
            result["text"] = text
            result["confidence"] = 0.7
            result["word_count"] = len(text.split())
            result["method"] = "easyocr"
            result["success"] = True
            return result
    except Exception:
        pass

    result["text"] = "[OCR: Could not extract text from this image. The image may be too low resolution or unclear.]"
    return result


def _tesseract_ocr(image_bytes: bytes, language: str = "eng"):
    """
    Run Tesseract OCR with full preprocessing pipeline.
    Returns (text, confidence_score).
    """
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
    import io as _io

    img = Image.open(_io.BytesIO(image_bytes))

    # Convert to RGB
    if img.mode not in ("RGB", "L", "RGBA"):
        img = img.convert("RGB")

    # Scale up small images
    w, h = img.size
    if max(w, h) < 800:
        scale = 800 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Convert to grayscale
    gray = img.convert("L")

    # Enhance contrast
    contrast = ImageEnhance.Contrast(gray).enhance(2.0)

    # Sharpen
    sharpened = contrast.filter(ImageFilter.SHARPEN)

    # OCR with detailed output
    data = pytesseract.image_to_data(
        sharpened,
        lang=language,
        config="--psm 6 --oem 3",
        output_type=pytesseract.Output.DICT
    )

    # Extract text and compute mean confidence
    words = []
    confidences = []
    for i, word in enumerate(data["text"]):
        conf = int(data["conf"][i])
        if conf > 10 and word.strip():
            words.append(word)
            confidences.append(conf)

    text = " ".join(words)
    avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0

    return text, avg_conf


def _easyocr_fallback(image_bytes: bytes):
    """
    Use EasyOCR as fallback for better multilingual support.
    """
    import easyocr
    import numpy as np
    from PIL import Image
    import io as _io

    img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    reader = easyocr.Reader(["en"], verbose=False)
    results = reader.readtext(img_array)

    return " ".join([res[1] for res in results if res[2] > 0.3])


def detect_document_orientation(image_bytes: bytes) -> int:
    """
    Detect if document is rotated. Returns degrees to rotate (0, 90, 180, 270).
    """
    try:
        import pytesseract
        from PIL import Image
        import io as _io

        img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        return osd.get("rotate", 0)
    except Exception:
        return 0


def auto_deskew_image(image_bytes: bytes) -> bytes:
    """
    Auto-deskew a scanned document image.
    Returns corrected image bytes.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import io as _io

        img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None:
            angles = []
            for line in lines[:20]:
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                if -45 < angle < 45:
                    angles.append(angle)

            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    h, w = img_array.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(img_array, M, (w, h),
                                             flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REPLICATE)

                    result_img = Image.fromarray(rotated)
                    buf = _io.BytesIO()
                    result_img.save(buf, format="PNG")
                    return buf.getvalue()

    except Exception:
        pass

    return image_bytes
