"""Pruebas automáticas M3: limpieza reutilizada, vectorizador sin leakage y métricas."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import CONFIGS, build_vectorizer, describe_errors, train_and_predict
from preprocessing import STOP, load_nlp, normalize, preprocess_text, preprocess_corpus


class PipelineTests(unittest.TestCase):
    def test_limpieza_html_y_url(self):
        text = normalize("<p>Strong &amp; safe</p> https://example.org")
        self.assertEqual(text, "strong & safe")

    def test_negaciones_se_conservan(self):
        nlp = load_nlp("fallback")
        self.assertIn("not", preprocess_text("Not a good match", nlp, "fallback").split())

    def test_consistencia_procesamiento_lote(self):
        nlp = load_nlp("fallback")
        rows = ["The cars are faster", "No way forward"]
        self.assertEqual(preprocess_corpus(rows, nlp, "fallback"),
                         [preprocess_text(s, nlp, "fallback") for s in rows])

    def test_experimentos_variacion_requisitos(self):
        self.assertTrue(any(c["ngram_range"] == (1, 2) for c in CONFIGS))
        self.assertGreater(len({c["max_features"] for c in CONFIGS}), 1)

    def test_vectorizador_no_ve_vocabulario_test(self):
        texts_train = ["apple banana fruit", "apple orange fruit", "banana orange fruit", "banana fruit"]
        labels = np.array(["World", "World", "Sports", "Sports"])
        texts_test = ["uniquetesttoken apple"]
        result = train_and_predict(texts_train, labels, texts_test, CONFIGS[0])
        self.assertNotIn("uniquetesttoken", result.vectorizer.vocabulary_)
        self.assertEqual(len(result.predictions), 1)

    def test_fit_transform_solo_en_train(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        calls = []
        actual_fit = TfidfVectorizer.fit_transform
        actual_transform = TfidfVectorizer.transform
        def spy_fit(instance, raw, *args, **kwargs):
            calls.append(("fit", tuple(raw)))
            return actual_fit(instance, raw, *args, **kwargs)
        def spy_transform(instance, raw, *args, **kwargs):
            calls.append(("transform", tuple(raw)))
            return actual_transform(instance, raw, *args, **kwargs)
        x_train = ["apple banana", "apple orange", "banana fruit", "orange fruit"]
        x_test = ["uniquetesttoken banana"]
        with patch.object(TfidfVectorizer, "fit_transform", spy_fit), patch.object(TfidfVectorizer, "transform", spy_transform):
            train_and_predict(x_train, np.array(["World", "World", "Sports", "Sports"]), x_test, CONFIGS[0])
        fit_docs = [docs for action, docs in calls if action == "fit"]
        test_calls = [docs for action, docs in calls if action == "transform" and docs == tuple(x_test)]
        self.assertEqual(fit_docs, [tuple(x_train)])
        self.assertEqual(len(test_calls), 1)

    def test_confusion_matrix_categorias(self):
        text, errs = describe_errors(["World", "World", "Sports"], ["World", "Sports", "Sports"])
        self.assertIn("World", text)
        self.assertEqual(errs[0]["errores"], 1)

    @unittest.skipUnless(__import__('importlib').util.find_spec("en_core_web_sm"),
                         "modelo SpaCy oficial no instalado en este entorno")
    def test_lemas_oficiales_si_estan_disponibles(self):
        nlp = load_nlp("spacy")
        tokens = preprocess_text("The cars were accelerating quickly", nlp, "spacy").split()
        self.assertIn("car", tokens)
        self.assertIn("accelerate", tokens)


if __name__ == "__main__":
    unittest.main()
