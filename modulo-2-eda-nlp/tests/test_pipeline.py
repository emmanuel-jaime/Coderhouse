"""Pruebas sin descargar modelos; las pruebas del modelo oficial son opcionales."""
import unittest
import spacy
import pandas as pd

from preprocess_eda import normalize, preprocess_text, ngrams, tokens_from_doc


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.blank("en")

    def test_html_y_urls(self):
        clean = normalize('&lt;b&gt;Hello&lt;/b&gt; https://example.org World')
        self.assertEqual(clean, "hello world")

    def test_elimina_rastro_de_agencia(self):
        self.assertNotIn("(ap)", normalize(' (AP) - Technology moves forward'))

    def test_conserva_negaciones(self):
        terms = preprocess_text('This is NOT a good idea, never!', self.nlp, 'fallback')
        self.assertIn('not', terms.split())
        self.assertIn('never', terms.split())
        self.assertNotIn('this', terms.split())

    def test_ngramas(self):
        terms = ngrams(pd.Series(['new york city', 'new york is big']), 2)
        self.assertEqual(terms[0], ('new york', 2))

    def test_sin_tokens_para_solo_ruido(self):
        self.assertEqual(preprocess_text('<div>and the</div>', self.nlp, 'fallback'), '')

    def test_pipeline_conducta_identica(self):
        text = 'The fastest players were running in London.'
        self.assertEqual(preprocess_text(text, self.nlp, 'fallback'),
                         ' '.join(tokens_from_doc(self.nlp(normalize(text)), 'fallback')))

    def test_lematizacion_oficial_si_esta_instalado(self):
        try:
            nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except OSError:
            self.skipTest('en_core_web_sm no instalado en este equipo')
        terms = preprocess_text('The cars were accelerating quickly.', nlp, 'spacy')
        self.assertIn('car', terms.split())
        self.assertIn('accelerate', terms.split())
        self.assertNotIn('cars', terms.split())


if __name__ == '__main__':
    unittest.main()
