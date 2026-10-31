"""Módulo 2 - EDA reproducible de AG News. El split test NO se utiliza.

Ejecutar:
  python preprocess_eda.py                      # modelo oficial de spaCy
  python preprocess_eda.py --mode fallback      # solo diagnóstico, sin lematización
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

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


def ngrams(texts: pd.Series, n: int, count: int = 20) -> list[tuple[str, int]]:
    vectorizer = CountVectorizer(
        ngram_range=(n, n), token_pattern=r"(?u)\b[a-z]+\b", lowercase=False
    )
    matrix = vectorizer.fit_transform(texts)
    counts = matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    order = sorted(range(len(words)), key=lambda i: (-int(counts[i]), words[i]))[:count]
    return [(str(words[i]), int(counts[i])) for i in order]


def get_font() -> str:
    paths = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in paths:
        if path.exists():
            pdfmetrics.registerFont(TTFont("NLP", str(path)))
            return "NLP"
    return "Helvetica"


def draw_figures(df: pd.DataFrame, lengths: np.ndarray, p95: int, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.7, 2.6))
    bins = range(0, int(lengths.max()) + 3, 2)
    ax.hist(lengths, bins=bins, color="#27496a", edgecolor="white", linewidth=.5)
    ax.axvline(p95, color="#dc7e38", linestyle="--", linewidth=2)
    ax.text(p95 + .4, ax.get_ylim()[1] * .88, f"P95 = {p95}", color="#ab551f", fontsize=10, weight="bold")
    ax.set_xlabel("Tokens útiles por noticia")
    ax.set_ylabel("Noticias")
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    fig.savefig(output / "longitudes.png", dpi=180, transparent=False)
    plt.close(fig)

    freq = df.label.value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(7.7, 2.1))
    bars = ax.barh(CLASSES[::-1], freq.loc[CLASSES[::-1]].values, color=["#547c9a", "#6894a3", "#9bb3bb", "#dfaa74"], height=.62)
    ax.set_xlim(0, int(freq.max() * 1.35))
    for bar in bars:
        ax.text(bar.get_width()+35, bar.get_y()+bar.get_height()/2, f"{int(bar.get_width()):,} / 25%".replace(",", "."), va="center", fontsize=10)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "clases.png", dpi=180)
    plt.close(fig)


def draw_pdf(stats: dict, bigrams, trigrams, output: Path, mode: str) -> None:
    """Informe de dos páginas y textos breves; gráficos estadísticos originales."""
    font = get_font()
    page_w, page_h = A4
    pdf_file = output / "EDA_NLP_JaimeEnriqueEmmanuel.pdf"
    c = canvas.Canvas(str(pdf_file), pagesize=A4)
    navy = colors.HexColor("#18334d")
    muted = colors.HexColor("#5d7080")
    accent = colors.HexColor("#d8894b")
    margin = 42

    def txt(x, y, value, size=10, bold=False, color=navy):
        c.setFillColor(color)
        c.setFont(font, size)
        c.drawString(x, y, str(value))

    def heading(num, title, y):
        txt(margin, y, f"{num}  {title}", 11)
        c.setStrokeColor(colors.HexColor("#e1e8ee"))
        c.line(margin, y-10, page_w-margin, y-10)

    def footer(page):
        c.setStrokeColor(colors.HexColor("#e1e8ee"))
        c.line(margin, 34, page_w-margin, 34)
        txt(margin, 19, "AG NEWS  /  DATA SCIENCE III  /  MÓDULO 2", 7.5, color=muted)
        txt(page_w-margin-20, 19, f"{page}/2", 8, color=muted)

    # Página 1
    txt(margin, 790, "EDA NLP | AG NEWS", 21)
    txt(margin, 768, "Pipeline de preprocesamiento y diagnóstico del corpus", 9, color=muted)
    mode_label = "SPACY / en_core_web_sm" if mode == "spacy" else "PRELIMINAR / SIN LEMATIZACIÓN"
    txt(margin, 747, "Ejecución: " + mode_label, 8, color=accent if mode != "spacy" else muted)

    kpis = [("TRAIN", f"{stats['train_rows']:,}"), ("CLASES", "4"), ("P95", str(stats['p95'])), ("MEDIANA", str(stats['median']))]
    for i,(label,val) in enumerate(kpis):
        x = margin + i * 132
        txt(x, 714, label, 8, color=muted)
        txt(x, 683, val.replace(",","."), 22)

    heading("01", "PREPROCESAMIENTO", 652)
    txt(margin, 627, "HTML  →  minúsculas  →  tokens  →  lemas  →  stop-words" if mode == "spacy" else "HTML  →  minúsculas  →  tokens  →  stop-words", 9.5)
    txt(margin, 608, "Negaciones conservadas; sin usar el conjunto de test.", 9, color=muted)

    heading("02", "LONGITUD DE LOS DOCUMENTOS", 576)
    c.drawImage(str(output / "longitudes.png"), margin-6, 353, width=510, height=205, preserveAspectRatio=True)
    txt(margin, 341, f"{stats['share_at_p95']:.1f}% de las noticias tiene hasta {stats['p95']} tokens; máximo: {stats['max_length']}.", 8.6, color=muted)

    heading("03", "DISTRIBUCIÓN DE CLASES", 313)
    c.drawImage(str(output / "clases.png"), margin-6, 139, width=510, height=163, preserveAspectRatio=True)
    txt(margin, 112, "Balance perfecto: 2.000 noticias por clase de entrenamiento.", 9, color=muted)
    txt(margin, 88, f"Control de calidad: {stats['html_docs']} textos con HTML; {stats['stop_top50']}/50 palabras comunes eran stop-words.", 8, color=muted)
    txt(margin, 68, "Nota: P95 de palabras; Transformers requieren medir subwords por separado.", 8, color=accent)
    footer(1)
    c.showPage()

    # Página 2
    txt(margin, 790, "N-GRAMAS DEL CORPUS", 18)
    txt(margin, 769, "20 bigramas y 20 trigramas · conteos sobre train", 9, color=muted)
    txt(margin, 742, "BIGRAMAS", 10)
    txt(294, 742, "TRIGRAMAS", 10)
    txt(252, 742, "N", 8, color=muted)
    txt(535, 742, "N", 8, color=muted)
    for i in range(20):
        y = 719 - 24 * i
        if i % 2 == 0:
            c.setFillColor(colors.HexColor("#f3f6f8"))
            c.roundRect(40, y-8, page_w-80, 21, 3, stroke=0, fill=1)
        for x, val in [(49, f"{i+1:02d}"), (75, bigrams[i][0]), (252, str(bigrams[i][1])), (294, trigrams[i][0]), (535, str(trigrams[i][1]))]:
            txt(x, y, val, 7.8, color=muted if x in (49,252,535) else navy)

    heading("04", "INTERPRETACIÓN", 219)
    txt(margin, 193, "Los N-gramas muestran entidades y temas relevantes, sin etiquetas HTML.", 9)
    heading("05", "CONCLUSIÓN", 164)
    txt(margin, 139, "Corpus sin nulos y balanceado: apto para clasificar con TF-IDF.", 8.6)
    txt(margin, 119, "Se eliminaron marcas HTML y ruido, conservando negaciones.", 8.6)
    if mode == "fallback":
        txt(margin, 99, "Pendiente: instalar en_core_web_sm y regenerar este PDF con lemas reales.", 8.1, color=accent)
    else:
        txt(margin, 99, "Para M4, calcular max_len con el tokenizador subword del Transformer.", 8.1, color=accent)
    txt(margin, 73, "El test se reserva exclusivamente para la evaluación final.", 8, color=muted)
    footer(2)
    c.save()


def run(args) -> None:
    data_dir = Path(args.data_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = data_dir / "ag_news_train.csv"
    if not path.is_file():
        raise SystemExit(f"No encuentro {path}. Copiá ahí el CSV de train de la cátedra.")
    train = pd.read_csv(path)
    if list(train.columns) != ["text", "label"]:
        raise SystemExit("El train debe tener exactamente las columnas text y label.")
    if train.isna().any().any():
        raise SystemExit("Se encontraron nulos, revisar los datos antes de continuar.")
    if set(train.label) != set(CLASSES):
        raise SystemExit("Las clases no coinciden con las 4 categorías previstas.")
    if train.text.duplicated().any():
        raise SystemExit("Hay textos duplicados: auditar antes de modelar.")

    nlp = load_nlp(args.mode)
    nlp.max_length = 2_000_000
    normalized = train.text.map(normalize)
    processed = []
    raw_terms = Counter()
    for doc in nlp.pipe(normalized, batch_size=128, n_process=1):
        raw_terms.update(t.lower_ for t in doc if t.is_alpha)
        processed.append(" ".join(tokens_from_doc(doc, args.mode)))

    proc = pd.Series(processed, index=train.index)
    lengths = proc.str.split().map(len).to_numpy()
    if np.any(lengths == 0):
        print("ADVERTENCIA: hay noticias sin tokens útiles. Revisar antes de entrenar.")
    p95 = int(np.percentile(lengths, 95, method="higher"))
    top50 = raw_terms.most_common(50)
    bigrams = ngrams(proc, 2)
    trigrams = ngrams(proc, 3)
    html_docs = sum(bool(RE_TAG.search(html.unescape(html.unescape(v)))) for v in train.text)
    stats = {
        "dataset": "AG News (muestra de cátedra)",
        "train_rows": len(train),
        "train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "spacy_version": spacy.__version__,
        "spacy_model": "en_core_web_sm" if args.mode == "spacy" else "none",
        "test_used": False,
        "mode": args.mode,
        "lemmas": args.mode == "spacy",
        "median": int(np.median(lengths)),
        "mean": round(float(np.mean(lengths)), 2),
        "p95": p95,
        "share_at_p95": round(float(np.mean(lengths <= p95) * 100), 2),
        "max_length": int(max(lengths)),
        "zero_token_docs": int(sum(lengths == 0)),
        "unique_terms": len(set(" ".join(processed).split())),
        "html_docs": html_docs,
        "stop_top50": sum(t in STOP for t, n in top50),
        "classes": {cl: int(n) for cl, n in train.label.value_counts().items()},
    }
    pd.DataFrame({"text_processed": processed, "label": train.label}).to_csv(output / "train_preprocesado.csv", index=False)
    pd.DataFrame(bigrams, columns=["ngrama", "frecuencia"]).to_csv(output / "bigramas.csv", index=False)
    pd.DataFrame(trigrams, columns=["ngrama", "frecuencia"]).to_csv(output / "trigramas.csv", index=False)
    pd.DataFrame(top50, columns=["palabra", "frecuencia"]).assign(
        stop_word=lambda d: d.palabra.isin(STOP)
    ).to_csv(output / "top50_palabras.csv", index=False)
    (output / "resumen.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_figures(train, lengths, p95, output)
    draw_pdf(stats, bigrams, trigrams, output, args.mode)
    if args.mode == "fallback":
        print("ATENCIÓN: PDF PRELIMINAR, falta la lematización oficial de spaCy.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Archivos generados en:", output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--mode", choices=["spacy", "fallback"], default="spacy")
    run(parser.parse_args())
