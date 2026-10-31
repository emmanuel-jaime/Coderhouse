"""Preprocesamiento reutilizado del Módulo 2 de Coderhouse (AG News).

Se mantienen las mismas expresiones regulares, stop-words y negaciones.
Con ``--mode spacy`` se utiliza lematización oficial con en_core_web_sm.
El modo fallback (sin lemas) es solo para reproducir una evaluación preliminar.
"""
from __future__ import annotations

import html
from pathlib import Path
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

CLASSES = ["World", "Sports", "Business", "Sci_Tech"]
KEEP_NEGATIONS = {"no", "nor", "not", "never", "nothing", "neither", "without"}
STOP = STOP_WORDS - KEEP_NEGATIONS
RE_TAG = re.compile(r"<[^>]+>")
# El proveedor agrega enlaces de cotización y duplicados de la firma periodística.
RE_QUOTE_LINK = re.compile(r"\([^)]*:\s*quote,\s*profile,\s*research\)", re.I)
RE_REPEATED_AGENCY = re.compile(
    r"\((?:AP|AFP|Reuters|Canadian Press)\)\.?\s*(?:AP|AFP|Reuters|Canadian Press)\s*[-–—:]",
    re.I,
)
RE_AGENCY_DATELINE = re.compile(r"\((?:AP|AFP|Reuters|Canadian Press)\)\s*[-–—:]", re.I)
RE_AGENCY_PREFIX = re.compile(r"(?:(?<=\.)|^)\s*(?:AP|AFP|Reuters|Canadian Press)\s*[-–—:]", re.I)
RE_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
RE_WS = re.compile(r"\s+")
RE_ARTIFACT = re.compile(r"(?i)\b(?:quot|nbsp|amp|lt|gt)\b")


def normalize(text: str) -> str:
    """Decodifica entidades, quita HTML y URL; conserva negaciones y términos."""
    text = html.unescape(html.unescape(str(text)))
    text = RE_TAG.sub(" ", text)
    text = RE_QUOTE_LINK.sub(" ", text)
    text = RE_REPEATED_AGENCY.sub(" ", text)
    text = RE_AGENCY_DATELINE.sub(" ", text)
    text = RE_AGENCY_PREFIX.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = text.lower().replace("\u2019", "'")
    return RE_WS.sub(" ", text).strip()


def load_nlp(mode: str):
    if mode == "spacy":
        try:
            return spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError as exc:
            raise SystemExit(
                "No se encuentra en_core_web_sm. Instalalo con: "
                "python -m spacy download en_core_web_sm; "
                "para una vista preliminar usá --mode fallback."
            ) from exc
    # Respaldo de diagnóstico SIN lematización, nunca presentarlo como SpaCy oficial.
    return spacy.blank("en")


def tokens_from_doc(doc, mode: str) -> list[str]:
    """Única implementación del filtrado para función y procesamiento masivo."""
    result = []
    for token in doc:
        if not token.is_alpha:
            continue
        term = token.lemma_.lower() if mode == "spacy" else token.lower_
        if term.isalpha() and term not in STOP and not RE_ARTIFACT.fullmatch(term):
            result.append(term)
    return result


def preprocess_text(text: str, nlp, mode: str = "spacy") -> str:
    """Normalización, tokenización, lematización y filtrado de stop-words."""
    if mode not in {"spacy", "fallback"}:
        raise ValueError("Modo no válido")
    return " ".join(tokens_from_doc(nlp(normalize(text)), mode))



def preprocess_corpus(texts, nlp, mode: str, batch_size: int = 128) -> list[str]:
    """Normaliza documentos uno a uno y procesa en lotes con el pipeline de M2."""
    if mode not in {"spacy", "fallback"}:
        raise ValueError("Modo no válido")
    normalized = (normalize(text) for text in texts)
    return [" ".join(tokens_from_doc(doc, mode))
            for doc in nlp.pipe(normalized, batch_size=batch_size, n_process=1)]
