# Evaluación final (SpaCy oficial)

**Modelo:** LogisticRegression con TF-IDF, B_uni_bigramas_10000.

**Exactitud test:** 88.95%.  **F1 macro test:** 0.8896.

**Dificultades por categoría:** Menor recall: Business (85.8%). Confusión más frecuente: Sci_Tech → Business (46 noticias).

Se eligió la configuración utilizando únicamente la validación interna de train. TF-IDF se ajustó sobre train y se aplicó transform sobre test; sin fuga de datos.
