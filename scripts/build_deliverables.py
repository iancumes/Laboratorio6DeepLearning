"""Genera notebook autocontenido, analisis, PDF de cuatro paginas y vista HTML.

Todos los numeros provienen de artifacts/summary.json; no hay resultados simulados.
"""
from pathlib import Path
import ast, json, sys, textwrap, base64, html, inspect
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import nbformat as nbf
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from src.pipeline import LABELS, FIG, create_figures, cache_training_seconds

REPO='https://github.com/iancumes/Laboratorio6DeepLearning'
NB=ROOT/'notebooks/Laboratorio6_TransferLearning_Cumes_Valladares.ipynb'
PDF=ROOT/'output/pdf/Laboratorio6_TransferLearning_Cumes_Valladares.pdf'
REFERENCES=[
 ('CIFAR-10, Krizhevsky (2009)','https://www.cs.toronto.edu/~kriz/cifar.html'),
 ('Torchvision: VGG-16 y pesos','https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html'),
 ('PyTorch: transfer learning','https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html'),
 ('PyTorch: autograd y requires_grad','https://docs.pytorch.org/docs/2.14/notes/autograd.html'),
 ('PyTorch: optimizadores y param_groups','https://docs.pytorch.org/docs/2.14/optim.html'),
 ('PyTorch: Module.train y Module.eval','https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html'),
 ('THOP: contador de MACs','https://github.com/Lyken17/pytorch-OpCounter'),
 ('PyTorch: memoria CUDA','https://docs.pytorch.org/docs/2.14/generated/torch.cuda.max_memory_allocated.html'),
 ('PyTorch: sincronizacion CUDA','https://docs.pytorch.org/docs/2.14/generated/torch.cuda.synchronize.html'),
 ('Torchvision: MobileNet V3 Small','https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html')]

RESEARCH='''
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
'''

METHOD='''
Se usaron 45,000 imágenes de entrenamiento, 5,000 de validación estratificada y las 10,000 de test oficial. Hay 4,500/500/1,000 imágenes por clase respectivamente. El experimento reducido usa 450 por clase (4,500 en total), tomadas únicamente del split de entrenamiento. Semilla 23236. El test no selecciona hiperparámetros: primero se fija el ganador de cada familia por el máximo F1 macro de validación, se restaura su checkpoint y luego se realiza una evaluación final.

La CNN usa 32×32, tres bloques de dos convoluciones con BN/ReLU, max-pooling, dropout y pooling global. Se normaliza con estadísticas calculadas solamente en el split de 45,000. VGG-16 conserva 32×32 (Resize bilineal explícito a 32, sin aumento de píxeles) y normalización ImageNet; conserva su clasificador original salvo la última capa de diez clases. La entrada pequeña está permitida por la restricción de cómputo del enunciado. 32→224 multiplica los píxeles por 49; 32→32 por 1. Interpolar no crea detalle adicional y puede perder la escala de las features de ImageNet.

Augmentation: la CNN usa crop con padding reflectante y flip horizontal; el desplazamiento del crop es compartido por minibatch y los flips son independientes. VGG usa una vista de entrenamiento almacenada con flip horizontal aleatorio de probabilidad 0.5, fijado por imagen y semilla. No se aumenta validación ni test porque se requiere una medida estable de generalización y no una distribución aleatoria de evaluación. No se usa flip vertical porque cambia la orientación natural de vehículos y animales.

Para reducir cómputo en CPU se cachean features[:17], que siempre están congeladas, y para FE las features completas. Los mapas del prefijo se almacenan en float16 y se convierten a float32 antes del sufijo en ambas estrategias. En FT se recalculan los bloques entrenables. FT3 cachea también la salida del bloque 4 congelado (float32). FT1 usó cache hasta B3. Se incluye el costo de cache utilizado en la gráfica de tiempo. CNN/FE/FT1 usaron 8 hilos; FT2/FT3 e inferencia, 12. El tiempo de entrenamiento no mantiene fijos los hilos.

Para entradas 32×32, el quinto max-pooling produce un mapa 1×1. `avgpool(7,7)` replica cada valor 49 veces. El primer Linear se factoriza exactamente sumando esos 49 pesos antes de multiplicar. Los pesos y gradientes originales se conservan, y todos se actualizan mediante SGD. Para contar MACs nominales y cronometrar inferencia se desactiva esta factorización. Por ello FLOPs nominales y tiempo observado de entrenamiento no describen exactamente el mismo camino de ejecución; el informe lo declara. La equivalencia se verificó en forward, backward y tres pasos SGD con momentum y weight decay (tolerancia 1e-10 en float64); ver artifacts/model_invariants.json.

La CNN se entrena tres épocas y VGG dos épocas por el presupuesto de 45 minutos. Se registra cada época, sin early stopping. El análisis corresponde a esta búsqueda breve, no al potencial de las arquitecturas completamente convergidas. En el experimento reducido se mantienen las mismas épocas e hiperparámetros: también se reduce el número de pasos de optimización; no se aísla completamente cantidad de datos de presupuesto de entrenamiento.
'''

def pct(x): return f'{100*x:.2f}%'
def paired_confusions(cm,classes,n=3):
    cm=np.asarray(cm)
    pairs=sorted([(int(cm[i,j]+cm[j,i]),classes[i],classes[j]) for i in range(10) for j in range(i+1,10)],reverse=True)
    return '; '.join(f'{a}/{b}: {v}' for v,a,b in pairs[:n])

def discussion(s):
    finals=s['finals']; win=max(finals,key=lambda k:finals[k]['test']['f1_macro'])
    order=sorted(finals,key=lambda k:finals[k]['test']['f1_macro'],reverse=True)
    delta=(finals[order[0]]['test']['f1_macro']-finals[order[1]]['test']['f1_macro'])*100
    drop={k:(finals[k]['test']['f1_macro']-s['reduced_tests'][k]['test']['f1_macro'])*100 for k in LABELS}
    most=max(drop,key=drop.get)
    cnn=s['winners']['cnn']; vgg=s['winners']['ft']; costratio=finals['ft']['profile']['flops_inference']/finals['cnn']['profile']['flops_inference']
    answer=[]
    answer.append(('Desempeño y costo',f'{LABELS[win]} obtuvo el mayor F1 macro de test ({pct(finals[win]["test"]["f1_macro"])}) y accuracy {pct(finals[win]["test"]["accuracy"])}. La ventaja sobre el segundo fue {delta:.2f} puntos de F1. VGG-16 FT tiene {vgg["params_total"]/1e6:.2f} M parámetros frente a {cnn["params_total"]/1e6:.3f} M de la CNN y {costratio:.1f} veces sus FLOPs de inferencia. El costo adicional se justifica si se prioriza ese desempeño; una única semilla y dos o tres épocas limitan la fuerza de la comparación.'))
    answer.append(('Relación entre desempeño y recursos',f'La CNN tiene menor costo de inferencia, almacenamiento y parámetros; ofrece la opción más económica. La gráfica F1/costo permite decidir si la ganancia observada de VGG compensa el salto de cómputo. La gráfica de tiempo incorpora el costo inicial de cache train/validación: ignorarlo favorecería artificialmente al transfer learning. No se presenta un cociente arbitrario F1/FLOPs como una medida universal de calidad.'))
    answer.append(('Transferencia y redimensionamiento','Las convoluciones preentrenadas detectan bordes, texturas y formas reutilizables en imágenes de animales y vehículos. Los bloques superiores son más específicos a ImageNet. El resize aproxima la escala esperada pero solo interpola los 1,024 píxeles originales. Pasar a 224 produce 50,176 píxeles por canal, 49 veces más, y aproximadamente ese crecimiento en costo convolucional, sin multiplicar en la misma proporción el costo del clasificador. 32×32 reduce cómputo pero puede perjudicar desempeño.'))
    ft=[r for r in s['runs'] if r['config']['model']=='ft']
    text='; '.join(f'{r["id"]} B{r["config"]["blocks"]}, LR backbone {r["config"]["backbone_lr"]:g}: F1 val {pct(r["best_val_f1"])}' for r in ft)
    best=max(ft,key=lambda r:r['best_val_f1'])
    h=best['history']; gap=h[-1]['val_loss']-h[-1]['train_loss']
    overfit=[r['id'] for r in s['runs'] if r['history'][-1]['train_loss']<r['history'][0]['train_loss'] and r['history'][-1]['val_loss']>r['history'][0]['val_loss']]
    trends=('Señal inicial compatible con sobreajuste en '+', '.join(overfit)+'. ') if overfit else 'Las nueve curvas reducen ambas pérdidas entre primera y última epoch; no se observa esa señal temprana de sobreajuste. '
    text+=f'. Ganó {best["id"]}. {trends}La diferencia final loss validación - entrenamiento fue {gap:.3f}; dropout y augmentation hacen que esas pérdidas no sean estrictamente comparables. FT2/FT3 usan batch 1024 frente a 512 de FT1; FT3 cambia también dropout, así que no se aíslan completamente estos efectos. Dos épocas VGG no aseguran convergencia ni descartan overfitting posterior. No se comprobó catastrophic forgetting: requiere medir retención de la tarea original o comparar representaciones.'
    answer.append(('Bloques, learning rate y sobreajuste',text))
    answer.append(('Cantidad de datos','Cambios de F1 (100% menos 10%): '+ '; '.join(f'{LABELS[k]} {drop[k]:+.2f} puntos' for k in LABELS)+f'. {LABELS[most]} presenta la mayor caída en esta ejecución. Una representación preentrenada suele ser útil con pocos datos y congelar limita grados de libertad; FT requiere LR bajo y control de validación. Aquí conservar épocas también reduce pasos, así que estas diferencias no pueden atribuirse exclusivamente al tamaño de la muestra.'))
    conf=' '.join(f'{LABELS[k]}: {paired_confusions(f["confusion_matrix"],s["data"]["classes"])}.' for k,f in finals.items())
    sets=[]
    for f in finals.values():
        cm=np.asarray(f['confusion_matrix'])
        pairs=sorted([(int(cm[i,j]+cm[j,i]),i,j) for i in range(10) for j in range(i+1,10)],reverse=True)
        sets.append({(i,j) for _,i,j in pairs[:3]})
    common=set.intersection(*sets)
    conf+= (' Pares compartidos entre los tres top-3: '+', '.join(s['data']['classes'][i]+'/'+s['data']['classes'][j] for i,j in sorted(common))+'.') if common else ' No hay un par común a los tres top-3.'
    conf+=' Las frecuencias difieren por modelo; filas reales, columnas predichas y 1,000 ejemplos por fila.'
    answer.append(('Matrices de confusión',conf))
    answer.append(('Despliegue',f'Para un servidor con GPU priorizaríamos {LABELS[win]} si la calidad observada es el objetivo; validaríamos primero su latencia y memoria en la GPU real, pues aquí solo se midió CPU. Para móvil elegiríamos la CNN propia por menor número de parámetros y operaciones. Como alternativa preentrenada consideraríamos MobileNetV3 Small, diseñada con bloques ligeros; su calidad y latencia en este dataset necesitarían un experimento adicional.'))
    return answer

def build_notebook(s):
    cells=[]
    def md(t): cells.append(nbf.v4.new_markdown_cell(inspect.cleandoc(t)))
    def code(t):
        # Las definiciones extraidas de AST conservan su indentacion. Los
        # cuatro bloques de codigo redactados como strings tienen una primera
        # linea sin sangria y necesitan el tratamiento de docstrings.
        if t.startswith(('REENTRENAR =','from IPython.display','cnn_train_transform =','from sklearn.metrics')):
            t=inspect.cleandoc(t)
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(t).strip()))
    best=max(s['finals'],key=lambda k:s['finals'][k]['test']['f1_macro'])
    md(f'''# Laboratorio 6: Transfer Learning y Fine-Tuning
    **CC3092 - Deep Learning y Sistemas Inteligentes** · UVG · 30 septiembre 2026

    **Ian Cumes 23236 · Javier Valladares 23045**

    ## Resumen de resultados
    Se completaron 9 configuraciones, 21 épocas de búsqueda y 3 reentrenamientos con 10% de datos.
    El mejor F1 macro de test fue **{pct(s['finals'][best]['test']['f1_macro'])}** con **{LABELS[best]}**.
    Todos los resultados son mediciones locales; FLOPs de entrenamiento son estimaciones.
    Se trabajó en CPU Intel i9-13900H, sin CUDA. VGG usa 32×32 y la CNN 32×32.
    Este entrenamiento de dos o tres épocas por configuración está limitado por el tiempo disponible.

    Repositorio: [{REPO}]({REPO})
    ''')
    md('''## 1. Entorno y reproducibilidad
    Crear un entorno con Python 3.11, ejecutar `pip install -r requirements.txt` y abrir este notebook.
    El notebook contiene las definiciones de la implementación. Por defecto carga los resultados
    auditables del repositorio; `REENTRENAR=True` vuelve a ejecutar el pipeline si se eliminan los
    resultados anteriores. Las imágenes, caches y checkpoints grandes se generan localmente.
    Para iniciar una búsqueda distinta, usar un directorio de artifacts nuevo, porque los IDs
    existentes se reutilizan. Para Colab, clonar el repositorio y establecer su raíz como cwd.
    ''')
    src=(ROOT/'src/pipeline.py').read_text(encoding='utf-8')
    tree=ast.parse(src); defs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    first=defs[0].lineno-1; lines=src.splitlines(); header='\n'.join(lines[:first])
    header=header.replace("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path.cwd()\nif not (ROOT / 'src').exists(): ROOT = ROOT.parent\nassert (ROOT / 'src/pipeline.py').exists(), 'Abrir notebook desde el repositorio o notebooks/'")
    code(header)
    md('## 2. Investigación de transfer learning en PyTorch\n'+RESEARCH+'\n\nFuentes oficiales consultadas al final del notebook.')
    md('## 3. Diseño experimental y preparación de datos\n'+METHOD)
    groups={
      'seed_everything':'### Configuración, semillas y registro de hardware',
      'load_data':'### Dataset, separación estratificada, estadísticas y ejemplos',
      'SmallCNN':'## 4. Definición de los tres modelos\n### CNN propia: tres bloques convolucionales',
      'ReplicatedSpatialLinear':'### Factorización exacta del primer Linear de VGG-16',
      'make_vgg':'### VGG-16: pesos ImageNet, congelamiento y sustitución de la última capa',
      'preprocess':'### Pipeline nativo y pipeline VGG con normalización ImageNet',
      'build_cache':'### Cache del prefijo congelado y una vista de entrenamiento',
      'metrics':'## 5. Entrenamiento y evaluación\n### Métricas macro y conteo de parámetros',
      'batch_inputs':'### Entradas por minibatch y forward de capas entrenables',
      'evaluate':'### Validación con eval y no_grad',
      'run_training':'### Entrenamiento, grupos de parámetros y mejor checkpoint',
      'default_configs':'### Nueve iteraciones e hiperparámetros',
      'resource_profile':'## 6. Recursos\n### MACs por capa, FLOPs y latencia sincronizada',
      'final_evaluation':'### Evaluación final única de test y matrices',
      'run_all':'## 7. Ejecución y experimento con 10% de datos',
      'create_figures':'### Gráficas y tablas reproducibles'}
    for node in defs:
        if node.name in groups: md(groups[node.name])
        start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
        code('\n'.join(lines[start:node.end_lineno]))
    code('''REENTRENAR = False
    summary_path = ART / 'summary.json'
    if REENTRENAR or not summary_path.exists():
        results = run_all()
    else:
        results = json.loads(summary_path.read_text(encoding='utf-8'))
        print('Resultados medidos cargados; ver artifacts/runs y evaluations para auditoría.')
    assert len(results['runs']) == 9
    assert all(len(r['history']) == r['config']['epochs'] for r in results['runs'])
    assert len(results['reduced_tests']) == 3
    ''')
    md('## 8. Exploración y resultados\n### Balance y estadísticas por canal')
    code('''from IPython.display import display, Image
    d = results['data']
    display(pd.DataFrame({'clase':d['classes'],'train oficial':d['class_counts_official'],
                          'train': [4500]*10,'validación':[500]*10,'test':[1000]*10,'train 10%':[450]*10}))
    display(pd.DataFrame({'canal':['R','G','B'],'media CIFAR-10':d['mean_official'],
                          'std CIFAR-10':d['std_official'],'media ImageNet':d['imagenet_mean'],
                          'std ImageNet':d['imagenet_std']}))
    display(Image(filename=str(FIG / 'examples.png')))
    ''')
    md('''Esperamos mayor dificultad en cat/dog por pelaje y postura, deer/horse por silueta,
    automobile/truck por ruedas y carrocería y airplane/ship por fondos similares a esta resolución.
    Estos son supuestos previos; las confusiones realmente medidas aparecen abajo.''')
    md('### Pipelines torchvision equivalentes y preset oficial\nEl entrenamiento usa la versión vectorizada `preprocess`; los flips de VGG se fijan al construir la cache.')
    code('''cnn_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(d['mean_fit'], d['std_fit'])])
    cnn_eval_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(d['mean_fit'], d['std_fit'])])
    vgg_eval_transform = transforms.Compose([transforms.Resize((SIZE,SIZE), antialias=True),
        transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])
    vgg_train_transform = transforms.Compose([transforms.Resize((SIZE,SIZE), antialias=True),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])
    print('Preset original ImageNet (referencia, no se usa a 224 en estos runs):')
    print(models.VGG16_Weights.IMAGENET1K_V1.transforms())
    ''')
    md('### Tabla de las nueve iteraciones: mejor epoch de validación')
    code("iterations = pd.read_csv(ART / 'iterations.csv')\ndisplay(iterations[['id','modelo','bloques','lr','backbone_lr','dropout','weight_decay','batch','epochs','threads']])\ndisplay(iterations[['id','mejor_epoch','val_accuracy','val_precision','val_recall','val_f1']].round(4))\ndisplay(iterations[['id','params_total','params_trainable','seconds','gpu_peak_mb','cpu_rss_mb','block4_cache']].round(2))")
    md('### Historial completo: loss y métricas de cada epoch')
    code("history = pd.concat([pd.DataFrame(r['history']).assign(iteracion=r['id']) for r in results['runs']], ignore_index=True)\ndisplay(history[['iteracion','epoch','train_loss','val_loss','val_accuracy','val_precision_macro','val_recall_macro','val_f1_macro']].round(4))\ndisplay(history[['iteracion','epoch','train_seconds','epoch_seconds','cumulative_seconds']].round(2))\ndisplay(Image(filename=str(FIG / 'loss_curves_9.png')))")
    md('### Rendimiento, recursos y reducción de datos')
    code("comparison = pd.read_csv(ART / 'comparison.csv')\ndisplay(comparison[['modelo','id','test_accuracy','test_precision_macro','test_recall_macro','test_f1_macro']].round(4))\ndisplay(comparison[['modelo','params_total','params_trainable','macs_inference','latency_ms','seconds_per_epoch','best_epoch','seconds_total']].round(3))\ndisplay(comparison[['modelo','cpu_peak_rss_mb','gpu_peak_mb','estimated_flops_training_nominal','test_10pct_accuracy','test_10pct_precision_macro','test_10pct_recall_macro','test_10pct_f1_macro']].round(4))\ndisplay(pd.DataFrame(results['cache']['seconds'].items(),columns=['cache','segundos']))")
    md('''**Convención de costo.** MACs y FLOPs de inferencia corresponden al forward original,
    sin factorización. El entrenamiento registrado usa cache y factorización exacta.
    La estimación nominal es `N * E * (2*MACs_forward + 4*MACs_capas_entrenables)`;
    supone backward ≈ dos forwards para las capas entrenables. Es una estimación conservadora
    sin optimizador, activaciones ni validación. El cómputo ejecutado es menor con cache;
    la caché se calcula una vez en train con una vista aumentada. No confundir FLOPs nominales
    con operaciones realmente ejecutadas o con tiempo medido.
    `gpu_peak_mb=null` significa que no hubo GPU CUDA; RSS de CPU no es memoria GPU.
    ''')
    for name,title in [('accuracy_time.png','### Accuracy de validación contra tiempo acumulado'),
                       ('f1_cost.png','### F1 de test contra FLOPs nominales de inferencia'),
                       ('data_fraction.png','### Efecto de reducir los datos a 10%'),
                       ('confusion_matrices.png','### Matrices de confusión de las configuraciones finales')]:
        md(title)
        if name=='confusion_matrices.png':
            code("for model_kind in LABELS:\n    display(Image(filename=str(FIG / f'confusion_{model_kind}.png')))")
        else: code(f"display(Image(filename=str(FIG / '{name}')))")
    md('## 9. Discusión y conclusiones\n'+'\n\n'.join('### '+a+'\n'+b for a,b in discussion(s)))
    md('## 10. Comprobaciones de integridad')
    code('''from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    split = np.load(ART / 'split_indices.npz')
    assert len(set(split['train']) & set(split['validation'])) == 0
    assert set(split['reduced']).issubset(set(split['train']))
    for collection in [results['finals'], results['reduced_tests']]:
        for k, e in collection.items():
            preds = np.load(ART / 'evaluations' / f"{e['id']}_predictions.npz")
            y_true, y_pred = preds['y_true'], preds['y_pred']
            assert len(y_true) == 10000
            assert np.asarray(e['confusion_matrix']).sum() == 10000
            assert np.all(np.asarray(e['confusion_matrix']).sum(axis=1) == 1000)
            p,r,f,_ = precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
            assert abs(f-e['test']['f1_macro']) < 1e-12
            assert abs(accuracy_score(y_true,y_pred)-e['test']['accuracy']) < 1e-12
    print('9 configuraciones, 6 evaluaciones de test, métricas y splits verificados.')
    ''')
    md('## Referencias\n'+'\n'.join(f'- [{label}]({url})' for label,url in REFERENCES)+'\n\n'+f'Repositorio: [{REPO}]({REPO})')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                               'language_info':{'name':'python','version':'3.11'}})
    nbf.validate(nb); NB.parent.mkdir(exist_ok=True); nbf.write(nb,NB)

def build_analysis(s):
    t='# Análisis del Laboratorio 6\n\nIan Cumes 23236 · Javier Valladares 23045\n\n'
    t+='## Metodología\n'+METHOD+'\n## Investigación\n'+RESEARCH
    t+='\n## Resultados\n\n'+pd.read_csv(ROOT/'artifacts/comparison.csv').to_markdown(index=False)
    t+='\n\n'+'\n\n'.join('## '+a+'\n'+b for a,b in discussion(s))
    t+='\n\n## Referencias\n'+'\n'.join(f'- [{a}]({b})' for a,b in REFERENCES)+f'\n\nRepositorio: {REPO}\n'
    (ROOT/'ANALISIS.md').write_text(t,encoding='utf-8')

def build_pdf(s):
    import matplotlib.pyplot as plt
    PDF.parent.mkdir(parents=True,exist_ok=True)
    fonts=Path('C:/Windows/Fonts')
    pdfmetrics.registerFont(TTFont('Arial',str(fonts/'arial.ttf')))
    pdfmetrics.registerFont(TTFont('ArialBold',str(fonts/'arialbd.ttf')))
    c=canvas.Canvas(str(PDF),pagesize=(595.28,841.89)); W,H=595.28,841.89
    c.setTitle('Laboratorio 6 - Transfer Learning - Cumes y Valladares')
    c.setAuthor('Ian Cumes 23236; Javier Valladares 23045')
    dark=HexColor('#153447'); teal=HexColor('#147d92'); gray=HexColor('#48616f')
    left=40; width=W-80; y=0
    def page(num,title):
        nonlocal y
        c.setFillColor(dark);c.rect(0,H-82,W,82,fill=1,stroke=0)
        c.setFillColor(white);c.setFont('ArialBold',17);c.drawString(left,H-35,title)
        c.setFont('Arial',9);c.drawString(left,H-56,'CC3092 | Ian Cumes 23236 · Javier Valladares 23045 | 30/09/2026')
        c.setFillColor(gray);c.setFont('Arial',8);c.drawString(left,25,'Laboratorio 6 · Transfer Learning y Fine-Tuning')
        c.drawRightString(W-left,25,f'{num} / 4'); y=H-105
    def heading(t):
        nonlocal y
        c.setFillColor(teal);c.setFont('ArialBold',11);c.drawString(left,y,t);y-=18
    def para(t,size=9.1,leading=12.3):
        nonlocal y
        c.setFillColor(dark);c.setFont('Arial',size)
        for paragraph in t.split('\n'):
            words=paragraph.split();line=''
            for word in words:
                cand=(line+' '+word).strip()
                if pdfmetrics.stringWidth(cand,'Arial',size)>width:
                    c.drawString(left,y,line);y-=leading;line=word
                else:line=cand
            if line:c.drawString(left,y,line);y-=leading
            y-=3
    def table(headers,rows,widths,size=8.1,row_h=21):
        nonlocal y
        x=left;c.setFillColor(dark);c.rect(left,y-row_h+5,sum(widths),row_h,fill=1,stroke=0)
        c.setFont('ArialBold',size);c.setFillColor(white)
        for h,w in zip(headers,widths):c.drawString(x+5,y-10,h);x+=w
        y-=row_h
        for i,row in enumerate(rows):
            if i%2==0:c.setFillColor(HexColor('#eef4f6'));c.rect(left,y-row_h+5,sum(widths),row_h,fill=1,stroke=0)
            c.setFillColor(dark);c.setFont('Arial',size);x=left
            for v,w in zip(row,widths):c.drawString(x+5,y-10,str(v));x+=w
            y-=row_h
        y-=13
    page(1,'Transferencia con un presupuesto de CPU')
    heading('1. Datos y protocolo')
    d=s['data']
    para('CIFAR-10: 60,000 imágenes RGB de 32×32, 10 clases balanceadas; 50,000 de train oficial y 10,000 de test. Split estratificado 45,000/5,000 (4,500/500 por clase), test intacto (1,000 por clase); reducción a 4,500 (450 por clase). Semilla 23236. Se elige el checkpoint y la configuración por F1 macro de validación antes de evaluar test una vez por modelo y fracción.')
    para('Media RGB del train oficial: '+', '.join(f'{v:.4f}' for v in d['mean_official'])+'; desviación: '+', '.join(f'{v:.4f}' for v in d['std_official'])+'. ImageNet: media 0.485/0.456/0.406; desviación 0.229/0.224/0.225. La CNN normaliza con estadísticas del split de 45k. CNN: crop/flip, BN y dropout; VGG: flips fijados por semilla y almacenados. Validación y test tienen transformaciones deterministas.')
    heading('2. Investigación de PyTorch [2-9]')
    para('vgg16(weights=VGG16_Weights.IMAGENET1K_V1) reutiliza pesos ImageNet; weights.transforms() define resize 256, crop 224 y normalización. features contiene cinco bloques; avgpool=AdaptiveAvgPool2d(7,7) adapta la salida; classifier tiene tres Linear. Se conserva el clasificador y se reemplaza solo la última capa por Linear(4096,10).')
    para('requires_grad=False congela parámetros; True descongela B5 (features[24:]) o B4+B5 (features[17:]). train()/eval() controlan dropout/BN; eval no desactiva gradientes. no_grad evita el grafo en evaluación/cache. param_groups asigna LR menor al backbone; SGD usa momentum 0.9 y la CNN AdamW. THOP profile(inputs, custom_ops) cuenta MACs; hooks Conv/Linear permiten desglosarlos. Un MAC equivale aquí a dos FLOPs; totales incluye congelados y entrenables solo requires_grad=True.')
    para('CUDA: synchronize antes/después del cronómetro; reset_peak_memory_stats y max_memory_allocated para bytes pico de tensores. Hardware real: Intel i9-13900H, 15.64 GiB RAM; Intel Iris Xe sin CUDA. CNN/FE/FT1: 8 hilos; FT2/FT3 e inferencia: 12. Latencia/RSS son CPU; GPU N/D. VGG usa 32×32, manteniendo los píxeles de CIFAR-10; 224×224 sería 49 veces. Dos o tres épocas por configuración limitan convergencia.')
    heading('3. Nueve iteraciones: mejor epoch por F1 de validación')
    rows=[]
    for r in s['runs']:
        b=r['history'][r['best_epoch']-1];co=r['config']
        rows.append([r['id'],co.get('blocks',f'w={co.get("width")}'),f'{co["lr"]:g}',f'{co.get("backbone_lr",0):g}' if co['model']=='ft' else '-',str(co['dropout']),str(r['best_epoch']),f'{b["val_accuracy"]*100:.2f}',f'{b["val_f1_macro"]*100:.2f}'])
    table(['ID','Bloques/width','LR clf','LR base','Drop','Ep','Acc %','F1 %'],rows,[42,101,61,66,47,37,61,100],size=8,row_h=18)
    para('CNN: 3 epochs; VGG: 2 epochs; batch CNN 256, VGG 512 (FT2/FT3: 1024). Weight decay 1e-4, excepto CNN3 y FE3: 5e-4. Precision, recall, parámetros, tiempos, RAM y métricas por epoch se conservan en el notebook y los CSV.',size=8.5,leading=11)
    heading('Ganadores: mejor epoch, memoria y cómputo estimado')
    resource_rows=[]
    for kind,run in s['winners'].items():
        profile=s['finals'][kind]['profile']
        nominal=profile['estimated_flops_train_per_image']*run['n_train']*len(run['history'])/1e12
        resource_rows.append([run['id'],str(run['best_epoch']),f'{run["cpu_peak_rss_mb"]:.0f}','N/D',f'{nominal:.2f}'])
    table(['Modelo','Mejor epoch','Pico RSS CPU MB','Pico GPU','Entr. nominal TFLOPs'],resource_rows,[65,90,135,75,150],size=8.4,row_h=18)
    assert y>45,('page1 overflow',y);c.showPage()

    page(2,'Rendimiento y costo computacional')
    heading('4. Configuraciones finales y recursos')
    rows=[]
    for k,r in s['winners'].items():
        p=s['finals'][k]['profile']
        rows.append([r['id'],f'{r["params_total"]/1e6:.3f}',f'{r["params_trainable"]/1e6:.3f}',f'{p["macs_inference"]/1e9:.3f}',f'{p["latency_ms"]:.2f}',f'{np.mean([h["train_seconds"] for h in r["history"]]):.1f}',f'{r["total_seconds"]:.1f}'])
    table(['Modelo','Param M','Entr. M','GMAC/img','CPU ms','s/epoch','Total s'],rows,[58,77,76,92,68,72,72],size=8.5,row_h=19)
    rows=[]
    for k,r in s['winners'].items():
        p=s['finals'][k]['profile']; nominal=p['estimated_flops_train_per_image']*r['n_train']*len(r['history'])/1e12
        rows.append([r['id'],str(r['best_epoch']),f'{r["cpu_peak_rss_mb"]:.0f}','N/D',f'{nominal:.2f}'])
    para('Inferencia original: batch 1, cinco warm-ups y 30 repeticiones. FLOPs=2×MACs. Estimación nominal: N×E×(F_forward+2F_capas entrenables); backward solo en capas entrenables, sin optimizador ni validación. Cache y factorización exacta reducen el cómputo ejecutado respecto a esta estimación, conservando los pesos originales.',size=8.6,leading=11.3)
    ct=s['cache']['seconds'];extra=sum(s['cache'].get('block4_seconds',{}).values());para(f'Cache compartida: train {ct["train"]:.1f}s (una vista), validación {ct["validation"]:.1f}s, test {ct["test"]:.1f}s; cache B4 adicional {extra:.1f}s (FT3). Total de entrenamiento en tabla incluye validación/checkpoints; s/epoch es solo entrenamiento. RSS es máximo muestreado del proceso, no una medida de asignaciones CUDA.',size=8.6,leading=11.3)
    heading('5. Test oficial: métricas macro y efecto de reducir datos')
    rows=[]
    for k in LABELS:
        for frac,e in [('100%',s['finals'][k]),('10%',s['reduced_tests'][k])]:
            m=e['test']; rows.append([e['id'],frac,*[f'{m[z]*100:.2f}' for z in ['accuracy','precision_macro','recall_macro','f1_macro']]])
    table(['Modelo','Datos','Acc %','Precision %','Recall %','F1 %'],rows,[96,62,81,101,91,84],size=8.4,row_h=18)
    # Dos figuras completas con tipografia legible a media pagina.
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.45))
    for k,r in s['winners'].items():
        h=pd.DataFrame(r['history']);offset=cache_training_seconds(s,k,r['config'])
        axes[0].plot((h.cumulative_seconds+offset)/60,h.val_accuracy*100,'o-',label=r['id'])
        e=s['finals'][k];xx=e['profile']['flops_inference']/1e9; yy=e['test']['f1_macro']*100
        axes[1].scatter(xx,yy,s=45);axes[1].annotate(r['id'],(xx,yy),xytext=(4,7 if k!='fe' else -13),textcoords='offset points',fontsize=8)
    axes[0].set_xlabel('Tiempo acumulado (min)');axes[0].set_ylabel('Acc. validación (%)');axes[0].legend(fontsize=8);axes[0].set_ylim(0,100)
    axes[1].set_xscale('log');axes[1].set_xlabel('GFLOPs nominales / imagen');axes[1].set_ylabel('F1 macro test (%)');axes[1].set_ylim(0,100);axes[1].margins(x=.25)
    for ax in axes:ax.grid(alpha=.15);ax.tick_params(labelsize=8)
    fig.tight_layout();p=FIG/'report_resources.png';fig.savefig(p,dpi=200,bbox_inches='tight');plt.close(fig)
    iw,ih=ImageReader(str(p)).getSize();hh=width*ih/iw;c.drawImage(str(p),left,y-hh,width=width,height=hh);y-=hh+12
    para('Izquierda: incluye la cache de train/validación en VGG. Derecha: costo de inferencia nominal, no factorizado; las dos estrategias VGG tienen el mismo forward.',size=8,leading=10.5)
    assert y>43,('page2 overflow',y);c.showPage()

    page(3,'Curvas de aprendizaje y errores')
    heading('6. Loss de las configuraciones elegidas')
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.55),sharey=True)
    for ax,(k,r) in zip(axes,s['winners'].items()):
        h=pd.DataFrame(r['history']);ax.plot(h.epoch,h.train_loss,'o-',label='Train');ax.plot(h.epoch,h.val_loss,'s--',label='Val')
        ax.set_title(r['id'],fontsize=10);ax.set_xlabel('Epoch');ax.set_xticks(list(h.epoch));ax.tick_params(labelsize=8);ax.grid(alpha=.15)
    axes[0].set_ylabel('Entropía cruzada');axes[0].legend(fontsize=8)
    fig.tight_layout();p=FIG/'report_losses.png';fig.savefig(p,dpi=200,bbox_inches='tight');plt.close(fig)
    iw,ih=ImageReader(str(p)).getSize();hh=width*ih/iw;c.drawImage(str(p),left,y-hh,width=width,height=hh);y-=hh+17
    para('El notebook incluye las nueve curvas train/validación (tres por familia) y las pérdidas por epoch. La búsqueda breve permite observar aprendizaje inicial, pero no establecer el techo de calidad. Dropout y augmentation suelen elevar la loss de entrenamiento; una loss menor en validación no implica por sí sola fuga de datos.')
    heading('7. Confusiones más frecuentes en test')
    para('Se cuentan errores en ambas direcciones por par; hay 1,000 ejemplos reales por clase. Las matrices completas, con valores y etiquetas, están en el notebook.')
    for k,e in s['finals'].items():
        para(s['winners'][k]['id']+': '+paired_confusions(e['confusion_matrix'],d['classes']),size=9,leading=12)
    heading('8. Menos datos, misma configuración y número de épocas')
    fig,ax=plt.subplots(figsize=(7.1,2.5));xx=np.arange(3)
    aa=[s['finals'][k]['test']['f1_macro']*100 for k in LABELS];bb=[s['reduced_tests'][k]['test']['f1_macro']*100 for k in LABELS]
    ax.bar(xx-.18,aa,.36,label='45,000 imágenes');ax.bar(xx+.18,bb,.36,label='4,500 imágenes')
    ax.set_xticks(xx,[s['winners'][k]['id'] for k in LABELS]);ax.set_ylabel('F1 macro test (%)');ax.set_ylim(0,100);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
    for i,(a,b) in enumerate(zip(aa,bb)):
        ax.text(i-.18,a+1,f'{a:.1f}',ha='center',fontsize=8);ax.text(i+.18,b+1,f'{b:.1f}',ha='center',fontsize=8)
    fig.tight_layout();p=FIG/'report_fraction.png';fig.savefig(p,dpi=200,bbox_inches='tight');plt.close(fig)
    iw,ih=ImageReader(str(p)).getSize();hh=width*ih/iw;c.drawImage(str(p),left,y-hh,width=width,height=hh);y-=hh+15
    para(dict(discussion(s))['Cantidad de datos'],size=9,leading=12)
    assert y>43,('page3 overflow',y);c.showPage()

    page(4,'Interpretación y conclusiones')
    answers=dict(discussion(s))
    for title in ['Desempeño y costo','Relación entre desempeño y recursos','Transferencia y redimensionamiento','Bloques, learning rate y sobreajuste','Despliegue']:
        heading(title);para(answers[title],size=8.8,leading=11.5)
    heading('Referencias y entrega')
    para('[1] Krizhevsky, Learning Multiple Layers of Features from Tiny Images (2009). [2] Torchvision VGG-16. [3] PyTorch Transfer Learning tutorial. [4-6] PyTorch autograd, optim y Module. [7] THOP. [8-9] PyTorch CUDA memory/synchronize. [10] Torchvision MobileNetV3 Small. Enlaces completos consultados: notebook y ANALISIS.md.',size=8,leading=10.5)
    c.setFont('ArialBold',9);c.setFillColor(teal);c.drawString(left,y,'Repositorio Git (notebook ejecutado, código y resultados):');y-=16
    c.setFont('Arial',9);c.drawString(left,y,REPO);c.linkURL(REPO,(left,y-2,left+width,y+12),relative=0);y-=16
    para('Reproducir: pip install -r requirements.txt; python scripts/run_experiments.py. Las imágenes y pesos se descargan; los resultados medidos y predicciones están versionados. Una sola semilla, resolución reducida y dos o tres epochs limitan la generalización de las conclusiones.',size=8,leading=10.5)
    assert y>43,('page4 overflow',y);c.save()
    print('PDF listo:',PDF)

def build_readme(s):
    best=max(s['finals'],key=lambda k:s['finals'][k]['test']['f1_macro'])
    text=f'''# Laboratorio 6 - Transfer Learning y Fine-Tuning

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
Mejor F1 macro de test: **{pct(s['finals'][best]['test']['f1_macro'])} ({LABELS[best]})**.
El split es 45,000/5,000 estratificado; test contiene 10,000 imágenes y no se usa en la selección.
Hardware: Intel i9-13900H, 16 GB RAM; 8 hilos CNN/FE/FT1 y 12 FT2/FT3 e inferencia; no CUDA. Latencia CPU, GPU N/D.
VGG-16 usa 32×32 en ambas estrategias; CNN 32×32. Dos o tres epochs y una semilla limitan convergencia.

## Reproducir

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
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

Repositorio: {REPO}
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    s=json.loads((ROOT/'artifacts/summary.json').read_text(encoding='utf-8'))
    create_figures(s); build_notebook(s);build_analysis(s);build_pdf(s);build_readme(s)
    print('Entregables construidos.')
