"""
Image utilities for text extraction, captioning, and preprocessing.
Handles JPG, PNG, JPEG and scanned documents.
"""

import io
from typing import Optional


def preprocess_image_for_ocr(image_bytes: bytes) -> bytes:
    """
    Preprocess an image to improve OCR accuracy:
    - Convert to grayscale
    - Apply thresholding
    - Denoise
    Returns processed image bytes.
    """
    try:
        from PIL import Image, ImageFilter, ImageEnhance
        import io as _io

        img = Image.open(_io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Convert to grayscale
        gray = img.convert("L")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)

        # Apply sharpening
        sharpened = enhanced.filter(ImageFilter.SHARPEN)

        # Save to bytes
        buf = _io.BytesIO()
        sharpened.save(buf, format="PNG")
        return buf.getvalue()

    except Exception:
        return image_bytes   # Return original if preprocessing fails


def extract_text_from_image(image_bytes: bytes, ocr_enabled: bool = True) -> str:
    """
    Extract text from an image using OCR (pytesseract) or Gemini vision.
    Falls back gracefully if tesseract is not installed.
    """
    if not ocr_enabled:
        return "[OCR disabled. Enable in Settings to extract text from images.]"

    # Try pytesseract first (local OCR)
    try:
        return _ocr_with_tesseract(image_bytes)
    except Exception as tess_err:
        pass

    # Return placeholder if all OCR methods fail
    return "[Image uploaded. Text extraction requires Tesseract OCR. Image will be analyzed by AI vision.]"


def _ocr_with_tesseract(image_bytes: bytes) -> str:
    """Use pytesseract for local OCR."""
    try:
        import pytesseract
        from PIL import Image
        import io as _io

        # Preprocess for better results
        processed_bytes = preprocess_image_for_ocr(image_bytes)
        img = Image.open(_io.BytesIO(processed_bytes))

        # OCR with confidence
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text.strip() if text.strip() else "[No text detected in image]"

    except ImportError:
        raise RuntimeError("pytesseract not available")
    except Exception as e:
        raise RuntimeError(f"Tesseract OCR failed: {e}")


def caption_image(image_bytes: bytes, api_key: str, model: str = "gemini-1.5-flash") -> str:
    """
    Generate an AI caption/description for an image using Gemini vision.
    """
    try:
        import google.generativeai as genai
        import base64

        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model)

        # Encode image to base64
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect MIME type
        mime = _detect_image_mime(image_bytes)

        response = client.generate_content([
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime,
                            "data": b64
                        }
                    },
                    {
                        "text": (
                            "Describe this image in detail. If it contains text, tables, charts, "
                            "or diagrams, describe what they show. Be comprehensive and accurate."
                        )
                    }
                ]
            }
        ])

        return response.text if response.text else "Image analyzed but no description generated."

    except Exception as e:
        return f"[Image caption unavailable: {str(e)}]"


def _detect_image_mime(image_bytes: bytes) -> str:
    """Detect image MIME type from bytes header."""
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"   # Default


def resize_image_for_api(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """
    Resize an image if it exceeds max_size in either dimension.
    Keeps aspect ratio. Returns image bytes.
    """
    try:
        from PIL import Image
        import io as _io

        img = Image.open(_io.BytesIO(image_bytes))
        w, h = img.size

        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = _io.BytesIO()
        fmt = img.format or "PNG"
        img.save(buf, format=fmt)
        return buf.getvalue()

    except Exception:
        return image_bytes
