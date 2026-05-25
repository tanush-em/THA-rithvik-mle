"""Language detection returning ISO 639-1 codes.

Uses lingua-language-detector which is deterministic (no random seeds needed).
Falls back gracefully to 'en' on very short or ambiguous inputs.
"""
from __future__ import annotations

from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder


# Build a single module-level detector covering the languages present in the
# visible ticket set plus the most common support languages.
_SUPPORTED_LANGUAGES = [
    Language.ENGLISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.SPANISH,
    Language.PORTUGUESE,
    Language.CHINESE,
    Language.JAPANESE,
    Language.KOREAN,
    Language.ARABIC,
    Language.HINDI,
    Language.ITALIAN,
    Language.DUTCH,
    Language.RUSSIAN,
]

@lru_cache(maxsize=1)
def _detector():
    return (
        LanguageDetectorBuilder
        .from_languages(*_SUPPORTED_LANGUAGES)
        .with_minimum_relative_distance(0.1)
        .build()
    )


# Map lingua Language enum to ISO 639-1 code
_LINGUA_TO_ISO = {
    Language.ENGLISH: "en",
    Language.FRENCH: "fr",
    Language.GERMAN: "de",
    Language.SPANISH: "es",
    Language.PORTUGUESE: "pt",
    Language.CHINESE: "zh",
    Language.JAPANESE: "ja",
    Language.KOREAN: "ko",
    Language.ARABIC: "ar",
    Language.HINDI: "hi",
    Language.ITALIAN: "it",
    Language.DUTCH: "nl",
    Language.RUSSIAN: "ru",
}


def detect(text: str) -> str:
    """Detect the primary language of *text*. Returns an ISO 639-1 code.

    Defaults to 'en' when the text is too short, ambiguous, or the language
    is not in the supported set.
    """
    if not text or len(text.strip()) < 5:
        return "en"

    try:
        lang = _detector().detect_language_of(text)
        if lang is None:
            return "en"
        return _LINGUA_TO_ISO.get(lang, "en")
    except Exception:
        return "en"
