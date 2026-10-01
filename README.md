# Laboratorio 6 - Transfer Learning y Fine-Tuning

CC3092 · Universidad del Valle de Guatemala · 30 de septiembre de 2026.

**Ian Cumes 23236 · Javier Valladares 23045**

## Entregables

- `notebooks/Laboratorio6_TransferLearning_Cumes_Valladares.ipynb`: notebook completo con código y resultados guardados.
- `output/pdf/Laboratorio6_TransferLearning_Cumes_Valladares.pdf`: informe de cuatro páginas.
- `ANALISIS.md`: discusión completa y referencias.
- `src/pipeline.py`: implementación reutilizable y comentada.
- `artifacts/iterations.csv`, `comparison.csv`, `runs/`, `evaluations/`, `summary.json`: evidencia real, métricas y predicciones.
- `artifacts/figures/`: 50 ejemplos, nueve curvas, matrices y comparaciones.

## Resultado y alcance

Se ejecutaron nueve configuraciones: CNN de 3 epochs y VGG de 2 epochs y tres reentrenamientos con 10% de datos.
Mejor F1 macro de test: **68.60% (VGG-16 fine-tuning)**.
El split es 45,000/5,000 estratificado; test contiene 10,000 imágenes y no se usa en la selección.
Hardware: Intel i9-13900H, 16 GB RAM; 8 hilos CNN/FE/FT1 y 12 FT2/FT3 e inferencia; no CUDA. Latencia CPU, GPU N/D.
VGG-16 usa 32×32 en ambas estrategias; CNN 32×32. Dos o tres epochs y una semilla limitan convergencia.

## Reproducir

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_experiments.py
python scripts/build_deliverables.py
```

Python 3.11. Los runs terminados se reutilizan. Para reentrenar desde cero, guardar los resultados
previos y utilizar una carpeta `artifacts` nueva. Se requieren imágenes CIFAR-10 y pesos VGG-16
ImageNet; la cache y checkpoints se generan localmente y no se suben a Git por su tamaño.
El notebook carga los resultados versionados sin reentrenar, pero incluye todas las definiciones
y la opción `REENTRENAR`. Abrir desde la raíz o desde `notebooks/`.

## Medición y optimizaciones

Todos los datos de tablas son resultados locales. FLOPs de entrenamiento son **estimaciones
nominales**, forward + backward de capas entrenables. La latencia corresponde al modelo
original, sin factorización. El entrenamiento usa cache de bloques congelados y una
factorización matemáticamente equivalente del primer Linear cuando el mapa final es 1×1.
Todos los parámetros del clasificador original permanecen entrenables; se reemplaza solo
su última capa. La cache cuantiza a float16 y reconvierte a float32, igual para FE y FT.
Se reporta el tiempo de cache por separado y se incluye train/validación en la gráfica de tiempo.
Reducir datos con las mismas epochs también reduce pasos de optimización; no es un experimento
que aísle completamente cantidad de datos.

Repositorio: https://github.com/iancumes/Laboratorio6DeepLearning
