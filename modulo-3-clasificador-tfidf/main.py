"""Pre-entrega 3: TF-IDF + Regresión Logística en AG News.

Desde la carpeta del módulo:
  python main.py --data-dir data --mode spacy

Nota metodológica: las variantes se eligen en VALIDACIÓN (dentro de train).
El TEST se transforma y evalúa SOLO UNA VEZ al final; nunca se ajusta TF-IDF con test.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from preprocessing import CLASSES, load_nlp, preprocess_corpus, preprocess_text

SEED = 42
# Variamos explícitamente max_features y ngram_range (requisito de la cátedra).
CONFIGS = (
    {"nombre": "A_unigramas_5000", "max_features": 5000, "ngram_range": (1, 1)},
    {"nombre": "B_uni_bigramas_10000", "max_features": 10000, "ngram_range": (1, 2)},
    {"nombre": "C_uni_bigramas_20000", "max_features": 20000, "ngram_range": (1, 2)},
)


@dataclass
class FitResult:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    predictions: np.ndarray
    vocabulary_size: int


def read_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee exclusivamente los splits oficiales sin unirlos."""
    paths = (data_dir / "ag_news_train.csv", data_dir / "ag_news_test.csv")
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError(
            f"Copiá ag_news_train.csv y ag_news_test.csv en {data_dir.resolve()} "
            "(o indicá --data-dir)."
        )
    train_df, test_df = [pd.read_csv(p) for p in paths]
    for name, df in (("train", train_df), ("test", test_df)):
        if set(df.columns) != {"text", "label"}:
            raise ValueError(f"{name} debe tener columnas text y label")
        if df[["text", "label"]].isna().any().any():
            raise ValueError(f"{name} tiene valores nulos")
        if not set(df.label).issubset(set(CLASSES)):
            raise ValueError(f"{name} tiene categorías desconocidas")
    if set(train_df.text) & set(test_df.text):
        raise ValueError("Hay textos idénticos entre train y test: revisar fuga potencial")
    return train_df, test_df


def build_vectorizer(config: dict) -> TfidfVectorizer:
    """TF-IDF sobre texto tokenizado por M2, con matriz dispersa."""
    return TfidfVectorizer(
        max_features=config["max_features"],
        ngram_range=config["ngram_range"],
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        token_pattern=r"(?u)\b[a-z]+\b",
        dtype=np.float64,
    )


def train_and_predict(train_texts, train_labels, eval_texts, config: dict) -> FitResult:
    """La única función que realiza fit_transform: nunca con validación/test.

    Matrices eval siempre se crean mediante transform del vectorizador ya ajustado.
    """
    vectorizer = build_vectorizer(config)
    X_train = vectorizer.fit_transform(train_texts)  # Solo TRAIN.
    X_eval = vectorizer.transform(eval_texts)  # VALIDACIÓN o TEST, sin fit.
    classifier = LogisticRegression(
        C=2.0, max_iter=1200, solver="lbfgs", random_state=SEED
    )
    classifier.fit(X_train, train_labels)
    pred = classifier.predict(X_eval)
    return FitResult(vectorizer, classifier, pred, X_train.shape[1])


def plot_confusion(y_true, y_pred, output_file: Path, subtitle: str) -> None:
    """Matriz multiclase con conteos, ejes y etiquetas legibles."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    fig.patch.set_facecolor("white")
    img = ax.imshow(cm, cmap="Blues", vmin=0)
    for r in range(len(CLASSES)):
        for c in range(len(CLASSES)):
            ax.text(c, r, str(cm[r,c]), ha="center", va="center", fontsize=12,
                    weight="bold" if r == c else "normal",
                    color="white" if cm[r,c] > cm.max() * .55 else "#24364a")
    ax.set_xticks(range(4), CLASSES, rotation=24, ha="right")
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Categoría predicha", labelpad=10)
    ax.set_ylabel("Categoría real", labelpad=10)
    ax.set_title("AG News | Matriz de confusión", loc="left", fontsize=15, weight="bold", pad=16)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color="#62748a")
    fig.colorbar(img, ax=ax, shrink=.8, label="Noticias")
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def describe_errors(y_true, y_pred) -> tuple[str, list[dict]]:
    """Detecta mayor confusión por errores fuera de diagonal y peor recall."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    worst_idx = int(np.argmin(recalls))
    pairs = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                pairs.append({"real": CLASSES[i], "predicha": CLASSES[j], "errores": int(cm[i, j])})
    pairs.sort(key=lambda v: (-v["errores"], v["real"], v["predicha"]))
    if pairs:
        pair = pairs[0]
        text = (f"Menor recall: {CLASSES[worst_idx]} ({recalls[worst_idx]:.1%}). "
                f"Confusión más frecuente: {pair['real']} → {pair['predicha']} "
                f"({pair['errores']} noticias).")
    else:
        text = "No se observaron confusiones en el conjunto evaluado."
    return text, pairs[:8]


def run(args: argparse.Namespace) -> dict:
    if args.export_docs and args.mode != "spacy":
        raise ValueError("No se pueden exportar resultados preliminares a docs; usá --mode spacy")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df, test_df = read_splits(Path(args.data_dir))
    print(f"Corpus: {len(train_df)} train / {len(test_df)} test. Modo: {args.mode}.", flush=True)

    nlp = load_nlp(args.mode)
    print("Preprocesando train sin estadísticas del test...", flush=True)
    train_texts = preprocess_corpus(train_df.text, nlp, args.mode)
    print("Preprocesando test con las mismas reglas deterministas...", flush=True)
    test_texts = preprocess_corpus(test_df.text, nlp, args.mode)

    # El test nunca participa en la búsqueda ni en la selección de parámetros.
    x_fit, x_val, y_fit, y_val = train_test_split(
        train_texts, train_df.label.to_numpy(),
        test_size=0.20, stratify=train_df.label, random_state=SEED
    )
    experiments = []
    for config in CONFIGS:
        fitted = train_and_predict(x_fit, y_fit, x_val, config)
        row = {
            "configuracion": config["nombre"],
            "max_features": config["max_features"],
            "ngram_range": str(config["ngram_range"]),
            "vocabulario_real": fitted.vocabulary_size,
            "f1_macro_validacion": round(float(f1_score(y_val, fitted.predictions, average="macro")), 6),
            "accuracy_validacion": round(float(accuracy_score(y_val, fitted.predictions)), 6),
        }
        experiments.append(row)
        print(f"  {row['configuracion']}: F1 macro (val) = {row['f1_macro_validacion']:.4f}", flush=True)
    experiments_df = pd.DataFrame(experiments)
    experiments_df.to_csv(out / "comparacion_validacion.csv", index=False)
    # Desempate determinista: primero max F1, luego primera configuración definida.
    best_index = max(range(len(experiments)), key=lambda i: experiments[i]["f1_macro_validacion"])
    best = CONFIGS[best_index]
    print(f"Elegida (únicamente con validación): {best['nombre']}", flush=True)

    # Reentrenar con TODO train original, SIN usar test en el ajuste.
    final = train_and_predict(train_texts, train_df.label.to_numpy(), test_texts, best)
    y_true = test_df.label.to_numpy()
    result_report = classification_report(y_true, final.predictions, labels=CLASSES,
                                          target_names=CLASSES, digits=4, zero_division=0)
    result_dict = classification_report(y_true, final.predictions, labels=CLASSES,
                                        target_names=CLASSES, output_dict=True, zero_division=0)
    (out / "classification_report.txt").write_text(result_report, encoding="utf-8")
    pd.DataFrame(result_dict).transpose().to_csv(out / "classification_report.csv")
    cm = confusion_matrix(y_true, final.predictions, labels=CLASSES)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / "matriz_confusion.csv")
    subtitle = "TEST reservado | Preprocesamiento oficial SpaCy" if args.mode == "spacy" else "TEST reservado | PRELIMINAR sin lematización"
    plot_confusion(y_true, final.predictions, out / "matriz_confusion.png", subtitle)
    interpretation, errors = describe_errors(y_true, final.predictions)
    summary = {
        "dataset": "AG News (muestra de la cátedra)",
        "modo": args.mode,
        "lematizacion_oficial": args.mode == "spacy",
        "train_docs": int(len(train_df)),
        "validacion_interna_docs": int(len(y_val)),
        "test_docs": int(len(test_df)),
        "modelo": "LogisticRegression", "C": 2.0, "max_iter": 1200,
        "configuracion_elegida": best["nombre"],
        "max_features": best["max_features"],
        "ngram_range": list(best["ngram_range"]),
        "vocabulario_real_final": final.vocabulary_size,
        "accuracy_test": round(float(accuracy_score(y_true, final.predictions)), 6),
        "f1_macro_test": round(float(f1_score(y_true, final.predictions, average="macro")), 6),
        "interpretacion": interpretation,
        "confusiones_principales": errors,
        "uso_test_para_fit": False,
    }
    (out / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.predict_text:
        texto_limpio = preprocess_text(args.predict_text, nlp, args.mode)
        categoria = final.classifier.predict(final.vectorizer.transform([texto_limpio]))[0]
        print(f"EJEMPLO DE INFERENCIA: {args.predict_text!r} → {categoria}", flush=True)
    if args.export_docs:
        import shutil
        docs = Path(args.docs_dir)
        docs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "matriz_confusion.png", docs / "matriz_confusion.png")
        shutil.copy2(out / "classification_report.txt", docs / "classification_report.txt")
        shutil.copy2(out / "comparacion_validacion.csv", docs / "comparacion_validacion.csv")
        (docs / "analisis.md").write_text(
            "# Evaluación final (SpaCy oficial)\n\n"
            f"**Modelo:** LogisticRegression con TF-IDF, {best['nombre']}.\n\n"
            f"**Exactitud test:** {summary['accuracy_test']:.2%}.  "
            f"**F1 macro test:** {summary['f1_macro_test']:.4f}.\n\n"
            f"**Dificultades por categoría:** {interpretation}\n\n"
            "Se eligió la configuración utilizando únicamente la validación interna de train. "
            "TF-IDF se ajustó sobre train y se aplicó transform sobre test; sin fuga de datos.\n",
            encoding="utf-8"
        )
    print("\nREPORTE DE CLASIFICACIÓN (TEST):\n" + result_report, flush=True)
    print("ANÁLISIS: " + interpretation, flush=True)
    print("Salidas:", out.resolve(), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Carpeta con train/test CSV de la cátedra")
    parser.add_argument("--output-dir", default="outputs", help="Directorio generado (ignorarlo en Git)")
    parser.add_argument("--mode", choices=["spacy", "fallback"], default="spacy",
                        help="spacy es entrega oficial; fallback es preliminar sin lemas")
    parser.add_argument("--export-docs", action="store_true",
                        help="Copia gráfico, reporte y análisis al directorio versionable docs/ (solo SpaCy oficial)")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--predict-text", default=None,
                        help="Noticia cruda adicional para mostrar una predicción de categoría")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
