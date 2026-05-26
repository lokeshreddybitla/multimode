"""
AI utilities powered by Google Gemini.
Handles all AI inference: summarization, Q&A, entity extraction,
classification, sentiment analysis, study tools, and more.
"""

import json
import re
from typing import Dict, List, Any, Optional

from utils.pdf_utils import chunk_text
from utils.security import detect_prompt_injection


def get_gemini_client(api_key: str, model: str = "gemini-1.5-flash"):
    """Initialize and return a Gemini GenerativeModel client."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model)
    except ImportError:
        raise RuntimeError("google-generativeai is required. pip install google-generativeai")


def _call_gemini(prompt: str, api_key: str, model: str,
                 temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """
    Core Gemini API call with error handling.
    """
    try:
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model)

        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = client.generate_content(prompt, generation_config=config)

        if response and response.text:
            return response.text.strip()
        return "No response generated."

    except Exception as e:
        error_str = str(e)
        if "API_KEY_INVALID" in error_str or "API key not valid" in error_str:
            raise RuntimeError("Invalid API key. Please check your Gemini API key.")
        elif "RATE_LIMIT" in error_str or "quota" in error_str.lower():
            raise RuntimeError("API rate limit reached. Please wait a moment and try again.")
        elif "SAFETY" in error_str:
            raise RuntimeError("Content flagged by safety filters. Try rephrasing your question.")
        else:
            raise RuntimeError(f"AI error: {error_str[:200]}")


def detect_language(text: str, api_key: str, model: str) -> str:
    """Detect the language of a text snippet. Returns ISO language code."""
    try:
        prompt = f"""Detect the language of this text and respond with ONLY the ISO 639-1 
language code (e.g., 'en', 'es', 'fr', 'de', 'zh', 'ar').
Text: {text[:300]}
Response (code only):"""
        result = _call_gemini(prompt, api_key, model, temperature=0.1, max_tokens=10)
        return result.strip().lower()[:5]
    except Exception:
        return "en"


def summarize_document(text: str, api_key: str, model: str,
                       chunk_size: int = 3000) -> str:
    """
    Summarize a document, handling long texts via chunking.
    """
    if not text or len(text.strip()) < 50:
        return "Document appears to be empty or too short to summarize."

    # For short documents, summarize directly
    if len(text) <= chunk_size:
        prompt = f"""You are a professional document analyst. Provide a comprehensive summary of the following document.

Structure your response with:
1. **Overview** (2-3 sentences about what the document is)
2. **Key Points** (3-7 bullet points of the most important information)
3. **Main Topics** (list the primary subjects covered)
4. **Conclusion** (what the document concludes or recommends)

Document:
{text}

Summary:"""
        return _call_gemini(prompt, api_key, model, temperature=0.5, max_tokens=1024)

    # For long documents, chunk and summarize each chunk, then combine
    chunks = chunk_text(text, chunk_size=chunk_size)
    chunk_summaries = []

    for i, chunk in enumerate(chunks[:8]):  # Limit to 8 chunks for cost
        chunk_prompt = f"""Summarize the key information from this document section (section {i+1}/{len(chunks)}):

{chunk}

Provide a concise summary focusing on the most important facts, decisions, and insights:"""
        try:
            summary = _call_gemini(chunk_prompt, api_key, model, temperature=0.4, max_tokens=512)
            chunk_summaries.append(f"Section {i+1}: {summary}")
        except Exception:
            continue

    if not chunk_summaries:
        return "Could not generate summary. Please check your API key."

    # Combine chunk summaries
    combined = "\n\n".join(chunk_summaries)
    final_prompt = f"""Based on these section summaries of a long document, provide a unified, 
coherent summary with:
1. **Overview** (2-3 sentences)
2. **Key Points** (top 5-7 insights)
3. **Conclusion**

Section summaries:
{combined}

Unified Summary:"""

    return _call_gemini(final_prompt, api_key, model, temperature=0.5, max_tokens=1024)


def answer_question(question: str, context: str, history: List[Dict],
                    api_key: str, model: str) -> str:
    """
    Answer a question using retrieved document context.
    Refuses to hallucinate information not in the context.
    """
    # Security: check for prompt injection
    is_safe, reason = detect_prompt_injection(question)
    if not is_safe:
        return f"⚠️ {reason}"

    # Build conversation history string
    history_str = ""
    if history:
        recent = history[-6:]   # Last 3 exchanges
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

    prompt = f"""You are DocuMind AI, a document analysis assistant. Answer questions ONLY based on the provided document context.

RULES:
- Answer only from the provided context below
- If the answer isn't in the context, say "I couldn't find information about that in the uploaded documents."
- Cite which document or page your answer comes from when possible
- Be accurate, concise, and helpful
- Do not make up information
- If context is insufficient, suggest what additional documents might help

Document Context:
{context[:4000]}

Conversation History:
{history_str}

User Question: {question}

Answer:"""

    return _call_gemini(prompt, api_key, model, temperature=0.4, max_tokens=1024)


def extract_entities(text: str, api_key: str, model: str) -> Dict[str, List[str]]:
    """
    Extract named entities: people, organizations, dates, locations, numbers.
    Returns a dictionary of entity types to lists.
    """
    prompt = f"""Extract named entities from the following text. 
Return ONLY a valid JSON object with these keys:
- "people": list of person names
- "organizations": list of company/org names
- "locations": list of places
- "dates": list of dates/time references
- "numbers": list of important numbers/statistics
- "technologies": list of technologies/tools mentioned
- "key_terms": list of important domain-specific terms

Text:
{text[:3000]}

Return ONLY JSON, no explanation:"""

    try:
        result = _call_gemini(prompt, api_key, model, temperature=0.2, max_tokens=512)
        # Clean up potential markdown code blocks
        result = re.sub(r"```(?:json)?", "", result).strip()
        entities = json.loads(result)
        return entities
    except Exception:
        # Return empty dict on parse failure
        return {
            "people": [], "organizations": [], "locations": [],
            "dates": [], "numbers": [], "technologies": [], "key_terms": []
        }


def classify_document(text: str, api_key: str, model: str) -> Dict[str, str]:
    """
    Classify the document type, industry, and purpose.
    """
    prompt = f"""Classify this document. Return ONLY a valid JSON object with:
- "type": document type (e.g., "Research Paper", "Legal Contract", "Technical Manual", "Business Report", etc.)
- "industry": industry/domain (e.g., "Technology", "Healthcare", "Finance", "Education", etc.)
- "purpose": primary purpose in one sentence
- "audience": intended audience
- "confidence": confidence level ("High", "Medium", "Low")
- "description": 2-3 sentence description of what this document is

Text sample:
{text[:2000]}

Return ONLY JSON:"""

    try:
        result = _call_gemini(prompt, api_key, model, temperature=0.2, max_tokens=256)
        result = re.sub(r"```(?:json)?", "", result).strip()
        return json.loads(result)
    except Exception:
        return {
            "type": "Unknown", "industry": "General", "confidence": "Low",
            "purpose": "Could not classify", "audience": "General",
            "description": "Document classification failed."
        }


def analyze_sentiment(text: str, api_key: str, model: str) -> Dict[str, Any]:
    """
    Analyze sentiment and tone of a document.
    Returns score (0-1), label, and explanation.
    """
    prompt = f"""Analyze the sentiment and tone of this document. Return ONLY a valid JSON object with:
- "score": float between 0.0 (very negative) and 1.0 (very positive)
- "label": one of "Very Negative", "Negative", "Neutral", "Positive", "Very Positive"
- "tone": overall tone (e.g., "Formal", "Persuasive", "Informational", "Academic", "Casual")
- "explanation": 2-3 sentence explanation of the sentiment analysis
- "emotions": list of detected emotions (e.g., ["confident", "cautious", "optimistic"])

Text:
{text[:2000]}

Return ONLY JSON:"""

    try:
        result = _call_gemini(prompt, api_key, model, temperature=0.3, max_tokens=256)
        result = re.sub(r"```(?:json)?", "", result).strip()
        data = json.loads(result)
        # Ensure score is float between 0 and 1
        data["score"] = max(0.0, min(1.0, float(data.get("score", 0.5))))
        return data
    except Exception:
        return {
            "score": 0.5, "label": "Neutral", "tone": "Unknown",
            "explanation": "Sentiment analysis could not be completed.",
            "emotions": []
        }


def compare_documents(text_a: str, text_b: str, name_a: str, name_b: str,
                      api_key: str, model: str) -> Dict[str, Any]:
    """
    Compare two documents and return structured comparison results.
    """
    # Truncate each document
    truncated_a = text_a[:2000]
    truncated_b = text_b[:2000]

    prompt = f"""Compare these two documents and return ONLY a valid JSON object with:
- "overview": 3-4 sentence overview of both documents and their relationship
- "similarities": list of 3-6 similarity strings
- "differences": list of 3-6 key difference strings
- "recommendation": paragraph with actionable recommendation based on the comparison
- "similarity_score": float 0-1 representing how similar the documents are

Document A ({name_a}):
{truncated_a}

Document B ({name_b}):
{truncated_b}

Return ONLY JSON:"""

    try:
        result = _call_gemini(prompt, api_key, model, temperature=0.5, max_tokens=1024)
        result = re.sub(r"```(?:json)?", "", result).strip()
        return json.loads(result)
    except Exception:
        return {
            "overview": "Comparison could not be completed. Please try again.",
            "similarities": [], "differences": [],
            "recommendation": "Unable to generate recommendation.",
            "similarity_score": 0.5
        }


def generate_study_notes(text: str, api_key: str, model: str,
                         detail_level: str = "Standard") -> str:
    """
    Generate structured study notes from document text.
    """
    detail_instructions = {
        "Brief": "Create concise bullet-point notes. Maximum 300 words.",
        "Standard": "Create well-organized notes with headings and bullet points. 400-700 words.",
        "Comprehensive": "Create detailed study notes with explanations, examples, and key takeaways. 700-1200 words.",
    }

    instruction = detail_instructions.get(detail_level, detail_instructions["Standard"])

    prompt = f"""You are an expert study note creator. {instruction}

Format the notes with:
## Main Topic
### Sub-topic
- Key point
- Important detail

Include:
- Core concepts and definitions
- Important facts and statistics
- Key arguments or findings
- Terms to remember
- Summary at the end

Document content:
{text[:4000]}

Study Notes:"""

    return _call_gemini(prompt, api_key, model, temperature=0.5, max_tokens=1500)


def generate_flashcards(text: str, api_key: str, model: str,
                        num_cards: int = 10) -> List[Dict[str, str]]:
    """
    Generate Q&A flashcards from document content.
    """
    prompt = f"""Create exactly {num_cards} flashcards from this document. 
Return ONLY a valid JSON array where each item has:
- "question": clear, specific question
- "answer": concise but complete answer (2-4 sentences)
- "topic": topic/category this card belongs to

Make questions cover different aspects: definitions, concepts, facts, relationships, applications.

Document content:
{text[:3500]}

Return ONLY JSON array:"""

    try:
        result = _call_gemini(prompt, api_key, model, temperature=0.6, max_tokens=2048)
        result = re.sub(r"```(?:json)?", "", result).strip()
        cards = json.loads(result)
        return cards if isinstance(cards, list) else []
    except Exception:
        return []


def generate_quiz(text: str, api_key: str, model: str,
                  num_questions: int = 5, difficulty: str = "Medium") -> List[Dict]:
    """
    Generate multiple-choice quiz questions from document content.
    """
    difficulty_desc = {
        "Easy": "straightforward factual questions",
        "Medium": "questions requiring understanding and some inference",
        "Hard": "complex analytical questions requiring deep comprehension",
    }.get(difficulty, "mixed difficulty questions")

    prompt = f"""Create exactly {num_questions} multiple-choice quiz questions ({difficulty_desc}) 
from this document.

Return ONLY a valid JSON array where each item has:
- "question": the question
- "options": list of exactly 4 options (A, B, C, D formatted strings)
- "correct": the correct option (must match one of the options exactly)
- "explanation": why the answer is correct (1-2 sentences)
- "difficulty": "{difficulty}"

Document content:
{text[:3500]}

Return ONLY JSON array:"""

    try:
        result = _call_gemini(prompt, api_key, model, temperature=0.6, max_tokens=2048)
        result = re.sub(r"```(?:json)?", "", result).strip()
        questions = json.loads(result)
        return questions if isinstance(questions, list) else []
    except Exception:
        return []


def generate_key_insights(text: str, api_key: str, model: str) -> List[str]:
    """Extract the most important insights from a document."""
    prompt = f"""Extract the 5-8 most important insights or takeaways from this document.
Return ONLY a JSON array of strings, each being one insight.

Document:
{text[:3000]}

Return ONLY JSON array:"""

    try:
        result = _call_gemini(prompt, api_key, model, temperature=0.5, max_tokens=512)
        result = re.sub(r"```(?:json)?", "", result).strip()
        insights = json.loads(result)
        return insights if isinstance(insights, list) else []
    except Exception:
        return []
