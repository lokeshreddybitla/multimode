"""
Security utilities for the Multimodal Document Analyzer.
Handles validation, sanitization, and rate limiting.
"""

import re
import hashlib
from typing import Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "jpg", "jpeg", "png"}
ALLOWED_MIME_TYPES = {
    "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain", "image/jpeg", "image/png",
}

# Prompt injection patterns to detect and block
INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"forget\s+everything",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(?:an?\s+)?(?:evil|unrestricted|jailbroken)",
    r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|prompts)",
    r"system\s*:\s*you\s+are",
    r"<\s*script[^>]*>",
    r"javascript\s*:",
    r"eval\s*\(",
    r"exec\s*\(",
    r"__import__",
]

# HTML tags to strip from extracted text
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

def validate_api_key(key: str) -> bool:
    """Basic format validation for Gemini API key."""
    if not key:
        return False
    # Gemini keys start with "AIza" and are 39 chars
    key = key.strip()
    return bool(re.match(r'^AIza[0-9A-Za-z\-_]{35}$', key))


def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate an uploaded file for type, size, and safety.
    Returns (is_valid, error_message).
    """
    if uploaded_file is None:
        return False, "No file provided."

    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        size_mb = uploaded_file.size / (1024 * 1024)
        return False, f"File too large ({size_mb:.1f} MB). Maximum is 50 MB."

    # Check extension
    name = uploaded_file.name.lower()
    if "." not in name:
        return False, "File has no extension."

    ext = name.rsplit(".", 1)[-1]
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type '.{ext}' is not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    # Check for executable-like content in filename
    dangerous_patterns = [r"\.exe$", r"\.bat$", r"\.sh$", r"\.py$",
                          r"\.js$", r"\.php$", r"\.rb$", r"\.cmd$"]
    for pattern in dangerous_patterns:
        if re.search(pattern, name, re.IGNORECASE):
            return False, "Potentially dangerous file type rejected."

    # Check for path traversal in filename
    if ".." in name or "/" in name or "\\" in name:
        return False, "Invalid filename."

    return True, ""


def sanitize_text(text: str, max_length: int = 500_000) -> str:
    """
    Sanitize extracted text:
    - Remove HTML tags
    - Strip null bytes
    - Truncate to max_length
    - Remove dangerous patterns
    """
    if not text:
        return ""

    # Remove HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)

    # Remove null bytes and control characters (except newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Normalize excessive whitespace (preserve paragraph structure)
    text = re.sub(r" {3,}", "  ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Truncate
    if len(text) > max_length:
        text = text[:max_length] + "\n\n[Document truncated for processing...]"

    return text.strip()


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    """
    Check user input for prompt injection attempts.
    Returns (is_safe, reason).
    """
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False, f"Potential prompt injection detected. Please ask a legitimate question about your documents."
    return True, ""


def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content for duplicate detection."""
    return hashlib.sha256(content).hexdigest()


def is_duplicate_document(file_hash: str, existing_hashes: set) -> bool:
    """Check if a document with this hash has already been uploaded."""
    return file_hash in existing_hashes


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and injection.
    """
    # Remove path components
    filename = filename.replace("/", "_").replace("\\", "_").replace("..", "_")
    # Allow only safe characters
    filename = re.sub(r"[^a-zA-Z0-9._\-\s]", "_", filename)
    return filename[:200]   # Limit length


def redact_sensitive_info(text: str) -> str:
    """
    Lightly redact patterns that look like API keys or credentials
    from text that will be displayed in UI (not sent to AI).
    """
    # Redact things that look like API keys
    text = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_API_KEY]", text)
    # Redact email addresses in previews
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                  "[EMAIL]", text)
    return text
