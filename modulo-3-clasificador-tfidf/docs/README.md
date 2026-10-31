# Evidencia del clasificador

La imagen `ejemplo_matriz_PRELIMINAR_fallback.png` proviene de una **evaluación real de referencia sin lematización** y no debe confundirse con los resultados oficiales.

Para la entrega final, ejecutar:

```powershell
python main.py --mode spacy --export-docs
```

El script generará en esta carpeta `matriz_confusion.png`, `classification_report.txt`, `comparacion_validacion.csv` y `analisis.md`, con resultados finales usando el modelo oficial `en_core_web_sm`. Después de revisar esos resultados, se puede retirar el ejemplo preliminar.
