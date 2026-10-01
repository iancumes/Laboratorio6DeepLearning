# Análisis del Laboratorio 6

Ian Cumes 23236 · Javier Valladares 23045

## Metodología

Se usaron 45,000 imágenes de entrenamiento, 5,000 de validación estratificada y las 10,000 de test oficial. Hay 4,500/500/1,000 imágenes por clase respectivamente. El experimento reducido usa 450 por clase (4,500 en total), tomadas únicamente del split de entrenamiento. Semilla 23236. El test no selecciona hiperparámetros: primero se fija el ganador de cada familia por el máximo F1 macro de validación, se restaura su checkpoint y luego se realiza una evaluación final.

La CNN usa 32×32, tres bloques de dos convoluciones con BN/ReLU, max-pooling, dropout y pooling global. Se normaliza con estadísticas calculadas solamente en el split de 45,000. VGG-16 conserva 32×32 (Resize bilineal explícito a 32, sin aumento de píxeles) y normalización ImageNet; conserva su clasificador original salvo la última capa de diez clases. La entrada pequeña está permitida por la restricción de cómputo del enunciado. 32→224 multiplica los píxeles por 49; 32→32 por 1. Interpolar no crea detalle adicional y puede perder la escala de las features de ImageNet.

Augmentation: la CNN usa crop con padding reflectante y flip horizontal; el desplazamiento del crop es compartido por minibatch y los flips son independientes. VGG usa una vista de entrenamiento almacenada con flip horizontal aleatorio de probabilidad 0.5, fijado por imagen y semilla. No se aumenta validación ni test porque se requiere una medida estable de generalización y no una distribución aleatoria de evaluación. No se usa flip vertical porque cambia la orientación natural de vehículos y animales.

Para reducir cómputo en CPU se cachean features[:17], que siempre están congeladas, y para FE las features completas. Los mapas del prefijo se almacenan en float16 y se convierten a float32 antes del sufijo en ambas estrategias. En FT se recalculan los bloques entrenables. FT3 cachea también la salida del bloque 4 congelado (float32). FT1 usó cache hasta B3. Se incluye el costo de cache utilizado en la gráfica de tiempo. CNN/FE/FT1 usaron 8 hilos; FT2/FT3 e inferencia, 12. El tiempo de entrenamiento no mantiene fijos los hilos.

Para entradas 32×32, el quinto max-pooling produce un mapa 1×1. `avgpool(7,7)` replica cada valor 49 veces. El primer Linear se factoriza exactamente sumando esos 49 pesos antes de multiplicar. Los pesos y gradientes originales se conservan, y todos se actualizan mediante SGD. Para contar MACs nominales y cronometrar inferencia se desactiva esta factorización. Por ello FLOPs nominales y tiempo observado de entrenamiento no describen exactamente el mismo camino de ejecución; el informe lo declara. La equivalencia se verificó en forward, backward y tres pasos SGD con momentum y weight decay (tolerancia 1e-10 en float64); ver artifacts/model_invariants.json.

La CNN se entrena tres épocas y VGG dos épocas por el presupuesto de 45 minutos. Se registra cada época, sin early stopping. El análisis corresponde a esta búsqueda breve, no al potencial de las arquitecturas completamente convergidas. En el experimento reducido se mantienen las mismas épocas e hiperparámetros: también se reduce el número de pasos de optimización; no se aísla completamente cantidad de datos de presupuesto de entrenamiento.

## Investigación

| API / concepto | Propósito y parámetros relevantes |
|---|---|
| `torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)` | Carga la arquitectura y los pesos supervisados de ImageNet. `weights=None` inicializa sin pesos; `progress` controla la descarga y `dropout` modifica el clasificador. `DEFAULT` selecciona el mismo conjunto de pesos. |
| `weights.transforms()` | Pipeline de inferencia asociado a esos pesos: resize 256, crop central 224, valores [0,1] y normalización RGB con media (0.485,0.456,0.406), desviación (0.229,0.224,0.225). Aquí se adapta el resize a 32 por presupuesto de CPU, conservando la normalización. |
| `features`, `avgpool`, `classifier` | VGG-16 contiene 13 convoluciones organizadas en cinco bloques. `features` incluye convoluciones, ReLU y max-pooling. `AdaptiveAvgPool2d((7,7))` conserva las dimensiones de entrada del clasificador; los Linear son 25088→4096→4096→1000. Se reemplaza únicamente el último Linear por 4096→10. |
| `requires_grad` | Controla si autograd calcula gradientes para un parámetro. Se congela `features` completo; para FT se activan parámetros desde índice 24 (B5) o 17 (B4+B5). Los 119,586,826 parámetros del clasificador permanecen entrenables. |
| `model.train()` y `model.eval()` | Cambian el comportamiento de dropout y batch normalization. `eval()` por sí solo no desactiva autograd. Train permite regularización; evaluación usa BN estable y dropout desactivado. El prefijo congelado permanece en eval. |
| `torch.no_grad()` | Evita construir el grafo de gradientes durante inferencia y generación de cache. No se usa alrededor del sufijo entrenable de FT. |
| `param_groups` | El optimizador recibe listas con `params`, `lr` y opcionalmente `weight_decay`. Permiten LR del backbone menor que el clasificador, protegiendo la representación preentrenada. SGD usa `momentum=0.9`; la CNN usa AdamW. |
| `thop.profile(model, inputs=(x,))` | Estima MACs y parámetros a partir de una entrada concreta; `custom_ops` define contadores para módulos no estándar y `verbose=False` reduce logs. El costo depende de la forma de entrada. También se registra un contador de hooks para Conv/Linear por capa. |
| `torch.cuda.max_memory_allocated()` | Máximo de bytes ocupados por tensores CUDA desde `reset_peak_memory_stats()`. No equivale a toda la memoria reservada ni al consumo total del proceso. En CPU no existe esta medida: se registra `null` y RSS del proceso por separado. |
| `torch.cuda.synchronize()` | Espera que las operaciones CUDA terminen; debe rodear el tramo cronometrado para no medir solamente la cola asíncrona. La latencia aquí es de CPU, batch 1, cinco warm-ups y treinta repeticiones. |

Un **MAC** acumula un producto; con la convención usada, multiplicación y suma son dos operaciones: **FLOPs ≈ 2×MACs**. Se excluyen activaciones, pooling, BN y actualizaciones del optimizador del contador nominal. Los parámetros **totales** incluyen capas congeladas; los **entrenables** son aquellos con `requires_grad=True`. Congelar reduce backward y memoria de gradientes, pero el forward del backbone sigue siendo necesario sin cache.

## Resultados

| modelo             | id   |   params_total |   params_trainable |   macs_inference |   flops_inference |   latency_ms |   seconds_per_epoch |   best_epoch |   epochs |   seconds_total |   gpu_peak_mb |   cpu_peak_rss_mb |   estimated_flops_training_nominal |   test_accuracy |   test_precision_macro |   test_recall_macro |   test_f1_macro |   test_loss |   test_10pct_accuracy |   test_10pct_precision_macro |   test_10pct_recall_macro |   test_10pct_f1_macro |   test_10pct_loss |
|:-------------------|:-----|---------------:|-------------------:|-----------------:|------------------:|-------------:|--------------------:|-------------:|---------:|----------------:|--------------:|------------------:|-----------------------------------:|----------------:|-----------------------:|--------------------:|----------------:|------------:|----------------------:|-----------------------------:|--------------------------:|----------------------:|------------------:|
| CNN propia         | CNN1 |          72954 |              72954 |          9880192 |          19760384 |     0.507043 |             18.7476 |            3 |        3 |         58.0835 |           nan |           799.906 |                      8002955520000 |          0.6239 |               0.660594 |              0.6239 |        0.628404 |    1.05683  |                0.3865 |                     0.41434  |                    0.3865 |              0.379261 |           1.71482 |
| VGG-16 extractor   | FE2  |      134301514 |          119586826 |        432775168 |         865550336 |    16.7479   |             44.2383 |            2 |        2 |         91.8194 |           nan |          2687.96  |                    120947834880000 |          0.6732 |               0.673757 |              0.6732 |        0.672725 |    0.946265 |                0.5897 |                     0.591572 |                    0.5897 |              0.588871 |           1.20106 |
| VGG-16 fine-tuning | FT3  |      134301514 |          126666250 |        432775168 |         865550336 |    17.9908   |            102.133  |            2 |        2 |        208.903  |           nan |          2724.32  |                    131139993600000 |          0.6869 |               0.688625 |              0.6869 |        0.685986 |    0.902849 |                0.5786 |                     0.596888 |                    0.5786 |              0.579555 |           1.27758 |

## Desempeño y costo
VGG-16 fine-tuning obtuvo el mayor F1 macro de test (68.60%) y accuracy 68.69%. La ventaja sobre el segundo fue 1.33 puntos de F1. VGG-16 FT tiene 134.30 M parámetros frente a 0.073 M de la CNN y 43.8 veces sus FLOPs de inferencia. El costo adicional se justifica si se prioriza ese desempeño; una única semilla y dos o tres épocas limitan la fuerza de la comparación.

## Relación entre desempeño y recursos
La CNN tiene menor costo de inferencia, almacenamiento y parámetros; ofrece la opción más económica. La gráfica F1/costo permite decidir si la ganancia observada de VGG compensa el salto de cómputo. La gráfica de tiempo incorpora el costo inicial de cache train/validación: ignorarlo favorecería artificialmente al transfer learning. No se presenta un cociente arbitrario F1/FLOPs como una medida universal de calidad.

## Transferencia y redimensionamiento
Las convoluciones preentrenadas detectan bordes, texturas y formas reutilizables en imágenes de animales y vehículos. Los bloques superiores son más específicos a ImageNet. El resize aproxima la escala esperada pero solo interpola los 1,024 píxeles originales. Pasar a 224 produce 50,176 píxeles por canal, 49 veces más, y aproximadamente ese crecimiento en costo convolucional, sin multiplicar en la misma proporción el costo del clasificador. 32×32 reduce cómputo pero puede perjudicar desempeño.

## Bloques, learning rate y sobreajuste
FT1 B5, LR backbone 0.0001: F1 val 67.70%; FT2 B4+5, LR backbone 0.0001: F1 val 67.54%; FT3 B5, LR backbone 0.0005: F1 val 69.47%. Ganó FT3. Las nueve curvas reducen ambas pérdidas entre primera y última epoch; no se observa esa señal temprana de sobreajuste. La diferencia final loss validación - entrenamiento fue -0.027; dropout y augmentation hacen que esas pérdidas no sean estrictamente comparables. FT2/FT3 usan batch 1024 frente a 512 de FT1; FT3 cambia también dropout, así que no se aíslan completamente estos efectos. Dos épocas VGG no aseguran convergencia ni descartan overfitting posterior. No se comprobó catastrophic forgetting: requiere medir retención de la tarea original o comparar representaciones.

## Cantidad de datos
Cambios de F1 (100% menos 10%): CNN propia +24.91 puntos; VGG-16 extractor +8.39 puntos; VGG-16 fine-tuning +10.64 puntos. CNN propia presenta la mayor caída en esta ejecución. Una representación preentrenada suele ser útil con pocos datos y congelar limita grados de libertad; FT requiere LR bajo y control de validación. Aquí conservar épocas también reduce pasos, así que estas diferencias no pueden atribuirse exclusivamente al tamaño de la muestra.

## Matrices de confusión
CNN propia: cat/dog: 534; airplane/ship: 245; cat/frog: 226. VGG-16 extractor: cat/dog: 323; automobile/truck: 264; bird/frog: 189. VGG-16 fine-tuning: cat/dog: 325; automobile/truck: 246; bird/deer: 199. Pares compartidos entre los tres top-3: cat/dog. Las frecuencias difieren por modelo; filas reales, columnas predichas y 1,000 ejemplos por fila.

## Despliegue
Para un servidor con GPU priorizaríamos VGG-16 fine-tuning si la calidad observada es el objetivo; validaríamos primero su latencia y memoria en la GPU real, pues aquí solo se midió CPU. Para móvil elegiríamos la CNN propia por menor número de parámetros y operaciones. Como alternativa preentrenada consideraríamos MobileNetV3 Small, diseñada con bloques ligeros; su calidad y latencia en este dataset necesitarían un experimento adicional.

## Referencias
- [CIFAR-10, Krizhevsky (2009)](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Torchvision: VGG-16 y pesos](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html)
- [PyTorch: transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [PyTorch: autograd y requires_grad](https://docs.pytorch.org/docs/2.14/notes/autograd.html)
- [PyTorch: optimizadores y param_groups](https://docs.pytorch.org/docs/2.14/optim.html)
- [PyTorch: Module.train y Module.eval](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)
- [THOP: contador de MACs](https://github.com/Lyken17/pytorch-OpCounter)
- [PyTorch: memoria CUDA](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.max_memory_allocated.html)
- [PyTorch: sincronizacion CUDA](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.synchronize.html)
- [Torchvision: MobileNet V3 Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)

Repositorio: https://github.com/iancumes/Laboratorio6DeepLearning
