[ 🇨🇱 Español ] | [ 🇺🇸 [Read in English](README.en.md) ]

# EVALUACIÓN DE LA EFICIENCIA DE YOLOV8 EN GPU, CPU Y RASPBERRY PI: CONVOLUCIONES ESTÁNDAR VS SEPARABLES

> **Autor:** Pablo Vicente Reyes Pino
> **Programa:** Data Science
> **Profesor Guía:** Dr. Anthony D. Cho · **Profesor Revisor:** Carlos Muñoz
> **Institución:** Escuela de Ingeniería — Facultad de Ciencias, Ingeniería y Tecnología, Universidad Mayor (Santiago, Chile)
> **Documento de tesis:** Abril 2026 · **Defensa:** 24 de julio de 2026

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-2.13.1-FF6F00?logo=tensorflow&logoColor=white)
![keras_cv](https://img.shields.io/badge/keras__cv-0.9.0.1%20modificado-D00000)
![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-XNNPACK%20%2F%20Select%20TF%20Ops-FF6F00?logo=tensorflow&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi%204-despliegue%20embebido-A22846?logo=raspberrypi&logoColor=white)
![Dataset](https://img.shields.io/badge/dataset-COCO2017-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Repositorio del proyecto de tesis que compara **YOLOv8 con convoluciones estándar (Conv2D)** frente a **YOLOv8 con convoluciones separables en profundidad (SeparableConv2D)** en el módulo *Head*, entrenando ambos modelos sobre COCO2017 y midiendo su desempeño de detección y su costo de inferencia en **GPU, CPU y Raspberry Pi 4**.

---

## Índice

1. [Introducción](#1-introducción)
2. [Objetivos e hipótesis](#2-objetivos-e-hipótesis)
3. [Marco teórico](#3-marco-teórico)
4. [Metodología](#4-metodología)
5. [Desarrollo y resultados](#5-desarrollo-y-resultados)
6. [Conclusiones y futuras líneas de investigación](#6-conclusiones-y-futuras-líneas-de-investigación)
7. [Estructura del repositorio](#7-estructura-del-repositorio)
8. [Reproducción del experimento](#8-reproducción-del-experimento)
9. [Referencias](#9-referencias)

---

## 1. INTRODUCCIÓN

### 1.1 Motivación y justificación

La detección de objetos permite identificar y localizar elementos dentro de una imagen. Al integrarse con Deep Learning —y en particular con arquitecturas de la familia **YOLO (*You Only Look Once*)**— estos sistemas alcanzan una combinación de precisión y velocidad que habilita la toma de decisiones automatizada en tiempo real: asistencia y anticolisión vehicular, vigilancia, robótica, inspección industrial y control de tráfico.

La justificación del trabajo se apoya en tres ejes:

1. **Relevancia tecnológica de YOLOv8**, una arquitectura ampliamente adoptada que ofrece un equilibrio favorable entre precisión, eficiencia, flexibilidad y facilidad de uso.
2. **Uso en entornos operativos reales**, donde calidad y rapidez de detección son requisitos duros y evaluar el rendimiento del modelo es un paso previo necesario a cualquier implementación.
3. **Manejo de recursos**, ya que YOLOv8 está orientado a reducir el costo de cómputo respecto de otras alternativas de detección, lo que favorece su ejecución en sistemas con restricciones.

Sobre esa base se planteó **modificar YOLOv8 incorporando convoluciones separables en reemplazo de las convoluciones 2D tradicionales**, para analizar su impacto en precisión, velocidad de inferencia y eficiencia de cómputo, y determinar así la viabilidad real de despliegue en hardware reducido.

### 1.2 Descripción del problema

A pesar de los avances en redes convolucionales, el problema central para la adopción de la detección de objetos en aplicaciones de la vida real **no se limita a la calidad de la detección, sino a la viabilidad de ejecución en entornos con restricciones de cómputo**.

Los modelos de Deep Learning empleados habitualmente requieren una capacidad de procesamiento elevada, lo que dificulta su despliegue en plataformas embebidas o dispositivos de bajo consumo (por ejemplo, una Raspberry Pi), donde la memoria disponible, la potencia de CPU y el consumo energético son limitados. Se genera entonces un compromiso inevitable: al ejecutar modelos complejos en hardware reducido **la latencia aumenta y se compromete la operatividad del sistema**, aun cuando el modelo mantenga buen desempeño en infraestructura de alto rendimiento.

Esta brecha dificulta llevar soluciones de visión computacional a entornos reales de bajo costo y alta disponibilidad. En ese contexto surge la necesidad de evaluar una modificación de YOLOv8 mediante convoluciones separables, para analizar si es posible **mejorar la eficiencia computacional sin comprometer significativamente la capacidad de detección**.

![Figura 1: Ejemplo de detección de objetos (YOLO)](docs/images/figura1_ejemplo_yolo.png)

<p align="center"><em>Figura 1: Ejemplo de detección de objetos (YOLO). Ubica la detección de objetos dentro de la familia de tareas de visión computacional: segmentación semántica, clasificación + localización, detección de objetos y segmentación de instancias.</em></p>

### 1.3 Alcance del proyecto

El estudio abarcó la eficiencia de detección de objetos con modelos YOLOv8 sobre **dos plataformas de ejecución** (un computador de escritorio y una Raspberry Pi 4) y **dos configuraciones arquitectónicas** (implementación estándar con Conv2D e implementación modificada con convoluciones separables). La evaluación se realizó utilizando **COCO2017** como referencia, de manera comparable y reproducible.

Quedaron **fuera del alcance**: la optimización avanzada del modelo (cuantización, *pruning*, *fine-tuning* de recuperación de precisión) y las implementaciones en otros entornos de hardware.

---

## 2. OBJETIVOS E HIPÓTESIS

### 2.1 Objetivo general

> Analizar y evaluar el modelo YOLOv8 para la detección de objetos, implementándolo en un entorno de hardware con capacidades reducidas.

### 2.2 Objetivos específicos

- Analizar la arquitectura y las mejoras en las técnicas introducidas en YOLOv8.
- Desarrollar versiones del modelo YOLOv8, en su configuración clásica y modificada, que permitan su análisis y evaluación experimental.
- Evaluar el rendimiento de los modelos en términos de precisión, velocidad y eficiencia computacional.
- Comparar el rendimiento de las dos versiones de YOLOv8 en un hardware reducido.

### 2.3 Hipótesis de investigación

> Los modelos de Deep Learning adaptados para hardware reducido tendrán una precisión mayor que los modelos de Deep Learning tradicionales en función de los recursos disponibles, comparando métricas específicas de rendimiento.

> [!NOTE]
> Los resultados de la Sección 5 **no confirman la hipótesis en términos de calidad de detección**: la variante separable no supera al modelo estándar en mAP, IoU, precisión ni recall. Sí la confirman en términos de *eficiencia en función de los recursos disponibles*, que es donde la variante separable resulta claramente superior sobre hardware sin aceleración. Esta distinción se desarrolla en la Sección 6.

---

## 3. MARCO TEÓRICO

### 3.1 De características manuales a detectores de una sola pasada

La detección de objetos evolucionó desde descriptores manuales —**SIFT** (*Scale-Invariant Feature Transform*) y **HOG** (*Histogram of Oriented Gradients*)— hacia modelos de Deep Learning. Aquellos métodos capturaban rasgos geométricos locales y funcionaron como base para la clasificación, pero su rigidez frente a cambios de iluminación, rotaciones y escenas dinámicas impulsó la adopción de **redes neuronales convolucionales (CNN)**.

**AlexNet** (2012) demostró la eficacia de las CNN al ganar ImageNet, y fue seguida por **R-CNN**, **Fast R-CNN** y **Faster R-CNN**, que mejoraron precisión y eficiencia mediante propuestas de regiones. En 2015, **YOLO** introdujo la detección en una sola pasada sobre la imagen, subdividiéndola en una cuadrícula de regiones predictivas y logrando un equilibrio entre precisión y velocidad. Posteriormente se incorporaron **FPN** (*Feature Pyramid Networks*), **SSD** y arquitecturas basadas en *Transformers*, apoyadas en el avance de GPUs y TPUs.

En paralelo, la necesidad de ejecutar detección en dispositivos móviles y embebidos dio origen a las **convoluciones separables en profundidad**, popularizadas por **MobileNet** y **Xception**, y a técnicas de **poda de parámetros** y **cuantización de pesos**.

| Modelo | Ventajas | Limitaciones |
| :--- | :--- | :--- |
| **SIFT y HOG** | Efectivos para la detección de objetos básicos, especialmente en escenarios donde la variabilidad es limitada y las características son consistentes. | Limitaciones significativas frente a la variabilidad de las imágenes y el contexto; dependen en gran medida de la extracción manual de características. |
| **AlexNet y CNNs** | Las CNN, introducidas por AlexNet en 2012, aprenden representaciones de alto nivel directamente de los datos de imagen sin necesidad de características manuales, con mayor precisión y capacidad de generalización. | Requieren gran cantidad de datos y poder de cómputo para entrenarse eficientemente; la complejidad y el costo computacional son elevados. |
| **Modelos R-CNN** | Mejoraron precisión y eficiencia al integrar las etapas de propuestas de regiones y clasificación. Faster R-CNN introduce una Red de Propuestas de Regiones (RPN) que las genera de manera eficiente. | Complejidad arquitectónica que aumenta el tiempo de entrenamiento y la demanda computacional; requiere múltiples etapas en la detección. |
| **YOLO** | Cambió el paradigma al realizar la detección en una sola pasada, habilitando tiempo real; YOLOv3 y posteriores mejoraron precisión y eficiencia con técnicas avanzadas como convoluciones separables y redes de pirámides de características. | Dificultades en la detección precisa de objetos pequeños y en escenarios con muchas clases; sacrifica algo de precisión en favor de la velocidad. |
| **SSD y FPN** | SSD detecta en múltiples escalas y ratios de aspecto en una sola pasada; FPN emplea una arquitectura de pirámide de características para objetos de distintos tamaños y resoluciones. | Menor precisión frente a modelos más avanzados en ciertos escenarios; puede no manejar las variaciones de escala tan eficientemente como otros modelos. |
| **Transformers (DETR)** | Permiten manejar mejor las relaciones espaciales y contextuales en imágenes complejas, con mejoras significativas en precisión y eficiencia. | Relativamente nuevos en visión por computadora, con desafíos de optimización y adopción; su complejidad dificulta el entrenamiento y el despliegue. |

<p align="center"><em>Tabla 2.1: Comparación de modelos de detección de objetos en términos de ventajas y limitaciones.</em></p>

### 3.2 Arquitectura de YOLOv8

YOLOv8 (2023) es un detector *single-stage* basado en una red convolucional profunda que predice cajas delimitadoras y clases en un único ciclo. Su arquitectura se organiza en tres bloques: **Backbone**, **Neck** y **Head**.

![Figura 2: Arquitectura de YOLOv8 (Backbone-Neck-Head)](docs/images/figura2_arquitectura_yolov8.png)

<p align="center"><em>Figura 2: Modelo implementado de YOLOv8 estándar (Terven et al., 2023). Entrada 640×640×3, cuatro etapas de Backbone, fusión multiescala en el Neck y tres ramas de predicción desacopladas en el Head.</em></p>

#### Backbone

Extrae características jerárquicas desde la imagen de entrada y produce mapas de características de alto nivel.

| Componente | Función |
| :--- | :--- |
| **CSPDarknet** | Variante de Darknet que divide la red en dos rutas parciales (*Cross Stage Partial*), reduciendo redundancia de cómputo y mejorando el aprendizaje de características. |
| **Capas convolucionales (Conv2D)** | Aplican filtros sobre la entrada para detectar bordes, texturas y patrones. |
| **Batch Normalization** | Normaliza la salida de la capa previa, estabilizando el entrenamiento. |

#### Funciones de activación: SiLU

YOLOv8 utiliza **SiLU** (*Sigmoid Linear Unit*, también conocida como Swish), definida como:

$$\mathrm{SiLU}(x) = x \cdot \sigma(x)$$

donde $\sigma(x)$ es la función sigmoide:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Sus ventajas frente a ReLU o Leaky ReLU son:

- **Mejor flujo de gradientes:** reduce el problema del *vanishing gradient* en redes profundas.
- **Mayor expresividad:** conserva valores negativos útiles, permitiendo modelar relaciones más complejas.
- **Mejor rendimiento en visión:** mejora la precisión de detección y segmentación en modelos como YOLOv8.
- **Continuidad y suavidad:** no introduce puntos de discontinuidad, lo que estabiliza el entrenamiento.

#### Neck

Actúa como puente entre el Backbone y el Head, consolidando y refinando las características extraídas:

- **Fusión de características:** estructuras **FPN** (*Feature Pyramid Network*) y **PANet** (*Path Aggregation Network*) que combinan información de múltiples escalas, aprovechando tanto rasgos de bajo nivel como de alto nivel.
- **Refinamiento de características:** capas especializadas que descartan información irrelevante antes de pasar al Head.
- **Capas de concatenación (Concat):** integran mapas de distintos niveles de la red.
- **Upsampling:** reconstruye información espacial perdida en convoluciones y *pooling*, mediante interpolación y *transposed convolutions*.

#### Head

Genera las predicciones finales —clases y coordenadas de cajas— a partir de las características entregadas por el Neck. **Es el módulo que este trabajo modifica.**

- **Estrategia desacoplada (*Decoupled Head*):** separa la rama de clasificación de la rama de regresión de cajas, optimizando el rendimiento en escenarios complejos.
- **Predicción en tres niveles jerárquicos:**
  - `yolo_v8_head_1` — objetos de menor tamaño (80×80).
  - `yolo_v8_head_2` — objetos de tamaño intermedio (40×40).
  - `yolo_v8_head_3` — objetos de gran tamaño (20×20).
- **Mapas de predicción por escala:**
  - *Mapa de clasificación:* probabilidad de cada clase en cada celda de la cuadrícula de salida.
  - *Mapa de regresión de cajas:* coordenadas normalizadas $(x, y, w, h)$, donde $(x,y)$ es el centro de la caja y $(w,h)$ su ancho y alto.
- **Non-Maximum Suppression (NMS):** filtra detecciones redundantes conservando las de mayor confianza y eliminando aquellas cuyo IoU con una predicción de mayor confianza supere un umbral predefinido.
- **Post-procesamiento:** salida final con coordenadas ajustadas, puntuaciones de confianza y etiquetas de clase.

#### Componentes avanzados

| Módulo | Descripción |
| :--- | :--- |
| **Residual blocks** | Dos convoluciones 3×3 con *padding* 1, Batch Normalization, activación SiLU y **conexión residual** (suma elemento a elemento entre la entrada del bloque y la salida de una capa posterior), que mitiga el desvanecimiento del gradiente sin incrementar la complejidad de cálculo. |
| **C2f (*Cross Stage Partial*)** | Introduce conexiones parciales entre etapas y *shortcuts* que mejoran la propagación de la información; combina convoluciones 1×1 y 3×3 con normalización por lotes. |
| **ConvModule** | Bloque básico: convolución 1×1 o 3×3 → Batch Normalization → SiLU. Su diseño modular permite reutilización a lo largo de la red. |
| **DarknetBottleneck** | Variante del bloque residual adaptada a Darknet, con capas convolucionales y conexiones de salto. |
| **SPPF (*Spatial Pyramid Pooling – Fast*)** | Captura información multiescala combinando ventanas de *pooling* de distinto tamaño, integrando detalles finos y dependencias de gran escala a bajo costo. |

### 3.3 Conv2D vs. convoluciones separables

![Figura 3: Conv2D vs SeparableConv2D](docs/images/figura3_conv2d_vs_separable.png)

<p align="center"><em>Figura 3: Conv2D vs SeparableConv2D (diagrama comparativo).</em></p>

#### Convolución estándar (Conv2D)

Realiza la extracción espacial y la combinación de canales **en una sola operación** matemática indexada por un filtro 3D. Para un kernel de tamaño $k \times k$, con $C_{in}$ canales de entrada y $C_{out}$ filtros de salida, el número total de parámetros es:

$$P_{\text{conv estándar}} = k \times k \times C_{in} \times C_{out}$$

donde:

- $k$ es el tamaño del kernel (p. ej. $3\times3$ o $5\times5$), que define el área espacial sobre la cual se aplican los filtros en cada paso de la convolución.
- $C_{in}$ es el número de canales de entrada, es decir, la profundidad de la entrada (p. ej. $C_{in}=3$ para una imagen RGB).
- $C_{out}$ es el número de filtros utilizados, lo que determina la cantidad de canales en la salida.

#### Paso 1 — Convolución depthwise

Aplica **un filtro 2D independiente por cada canal de entrada**, capturando patrones espaciales sin mezclar canales:

$$P_{\text{depthwise}} = k \times k \times C_{in}$$

El número de parámetros se reduce **en un factor de $1/C_{out}$** respecto de la convolución estándar. Como contrapartida, al operar cada canal de forma aislada, la convolución depthwise **por sí sola no puede aprender interacciones entre canales**, lo que limita su capacidad de representación.

#### Paso 2 — Convolución pointwise

Para superar esa limitación se aplica una convolución $1\times1$ sobre la salida depthwise —una **proyección lineal** que conserva la resolución espacial de los mapas de características y combina la información entre canales:

$$P_{\text{pointwise}} = C_{in} \times C_{out}$$

#### Costo combinado y factor de reducción

El costo total de la convolución separable es la **suma** de ambas etapas, en lugar del producto:

$$P_{\text{separable}} = \underbrace{k^2 \cdot C_{in}}_{\text{depthwise}} + \underbrace{C_{in} \cdot C_{out}}_{\text{pointwise}}$$

lo que da un factor de reducción de:

$$R = \frac{P_{\text{separable}}}{P_{\text{conv estándar}}} = \frac{k^2 C_{in} + C_{in} C_{out}}{k^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{k^2}$$

Para un kernel $3\times3$ y $C_{out}$ grande, $R \approx 1/9$: el costo aritmético crece **de forma aditiva** con el tamaño del kernel en lugar de multiplicativa.

![Figura 4: Ejemplo de procesos convolución estándar vs convolución separable](docs/images/figura4_procesos_convolucion.png)

<p align="center"><em>Figura 4: Ejemplo de procesos convolución estándar vs convolución separable. Arriba, el filtro 3D único que recorre simultáneamente espacio y canales; abajo, la descomposición en filtrado por canal (depthwise) seguido de la combinación entre canales (pointwise).</em></p>

#### Ventajas de las convoluciones separables

- **Reducción del número de parámetros:** las convoluciones estándar aplican un filtro 3D costoso; las separables lo descomponen en filtros 2D por canal más una combinación $1\times1$, manteniendo la capacidad de captura de características con menos parámetros.
- **Eficiencia computacional:** menos parámetros implican menos operaciones matemáticas en entrenamiento e inferencia, con una ejecución más rápida y eficiente en memoria y potencia de procesamiento.
- **Mejor rendimiento en dispositivos con recursos limitados:** al reducir la complejidad del modelo es posible ejecutar redes más profundas en plataformas con capacidad de procesamiento acotada.

> La arquitectura base de YOLOv8 **no** incorpora convoluciones separables de forma nativa; su integración es una variante experimental no oficial, y es exactamente la adaptación que este trabajo implementa y mide.

### 3.4 Selección de la plataforma embebida: Raspberry Pi 4 vs. ESP32-CAM

| Criterio | Raspberry Pi 4 | ESP32-CAM |
| :--- | :--- | :--- |
| **Procesador** | Quad-core ARM Cortex-A72 @ 1.5 GHz | ESP32-D0WDQ6, 2 núcleos @ 240 MHz |
| **Memoria** | 1 / 2 / 4 / 8 GB RAM | 520 KB RAM + 4 MB Flash |
| **Compatibilidad con Deep Learning** | Ejecuta modelos de detección; admite Intel Neural Compute Stick y Google Coral TPU | Tareas básicas de reconocimiento de imágenes |
| **Resultado en la fase de prueba** | Ejecutó YOLOv8 correctamente | **El modelo no pudo cargarse en memoria** |

Durante la fase de prueba se intentó ejecutar YOLOv8 en la ESP32-CAM, pero sus limitaciones de RAM y procesamiento impidieron cargar el modelo en la memoria del dispositivo. La Raspberry Pi 4 permite implementar YOLOv8 con un balance entre costo de cómputo y portabilidad, y por eso fue la plataforma embebida seleccionada.

---

## 4. METODOLOGÍA

### 4.1 Diseño de investigación

El diseño es **cuantitativo comparativo**: se busca medir y comparar el rendimiento de YOLOv8 estándar y YOLOv8 con convoluciones separables en términos de precisión, velocidad y costo de cómputo, mediante la recopilación de datos numéricos y su análisis estadístico. La comparación se realizó bajo un **mismo protocolo experimental**, cubriendo las etapas de preparación de datos, entrenamiento, validación e inferencia.

![Figura 5: Proceso de entrenamiento/validación](docs/images/figura5_proceso_entrenamiento.png)

<p align="center"><em>Figura 5: Proceso de entrenamiento/validación.</em></p>

Los pilares metodológicos son:

- **Diseño experimental comparativo:** evaluación objetiva de dos versiones de YOLOv8 (Conv2D estándar vs. convoluciones separables) bajo un protocolo idéntico de entrenamiento e inferencia.
- **Desarrollo de los modelos:** adaptación arquitectónica focalizada en el módulo **Head**, sustituyendo capas estándar por bloques separables para reducir la complejidad computacional.
- **Entrenamiento con COCO2017:** uso del *dataset* estándar bajo la misma preparación de datos, la misma división Train/Val y un ajuste homogéneo de hiperparámetros.
- **Métricas de evaluación:** medición multidimensional que balancea la calidad de detección (mAP, IoU, precisión, recall) frente a la eficiencia (velocidad de inferencia y recursos).
- **Validación multiplataforma (GPU, CPU y Raspberry Pi):** análisis del *trade-off* precisión vs. velocidad para determinar la viabilidad real de despliegue en hardware de bajo consumo.

### 4.2 Dataset: COCO2017

**COCO** (*Common Objects in Context*) contiene imágenes con múltiples objetos interactuando en entornos reales. En su versión 2017 reúne aproximadamente **330.000 imágenes** (más de 200.000 anotadas) y cerca de **1,5 millones de instancias de objetos anotadas**, organizadas en **80 categorías de objetos** contables y 91 categorías de *stuff* (agua, césped, cielo) que describen el entorno.

Para este trabajo se empleó la partición estándar de la literatura:

| Conjunto | Imágenes | Resolución de entrada | Formato de anotación | Categorías |
| :--- | ---: | :---: | :--- | ---: |
| **Train** | 118.287 | 640×640 | $(c_x, c_y, w, h)$ — formato YOLOv8 | 80 |
| **Val** | 5.000 | 640×640 | $(c_x, c_y, w, h)$ — formato YOLOv8 | 80 |

<p align="center"><em>Tabla 1: Base de datos COCO2017.</em></p>

#### Transformación de coordenadas

Las anotaciones de COCO definen las cajas como $(x_0, y_0, w, h)$, donde $(x_0,y_0)$ es la esquina superior izquierda y $(w,h)$ el ancho y alto en píxeles. Para compatibilizarlas con el formato basado en centro que usa YOLOv8, $(c_x, c_y, w, h)$, se aplicó:

$$c_x = x_0 + \frac{w}{2}, \qquad c_y = y_0 + \frac{h}{2}$$

Cuando la anotación de origen viene en formato de esquinas $(x_0, y_0, x_1, y_1)$, la conversión es:

$$w = x_1 - x_0, \qquad h = y_1 - y_0, \qquad c_x = \frac{x_0 + x_1}{2}, \qquad c_y = \frac{y_0 + y_1}{2}$$

Ambos modelos comparten exactamente el mismo flujo de preparación de datos.

### 4.3 Implementación del modelo

Ambos modelos se implementaron en **TensorFlow / Keras** (`tensorflow==2.13.1`) sobre `keras_cv`, partiendo del mismo *backbone*:

```python
stg = tf.distribute.MirroredStrategy()

with stg.scope():
    backbone = YOLOV8Backbone.from_preset("yolo_v8_xs_backbone", include_rescaling=True)
    YOLOV8_model = YOLOV8Detector(
        num_classes=num_classes,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=5,
    )

    optimizer = Adam(learning_rate=0.0007, weight_decay=0.0009, global_clipnorm=10.0)

    YOLOV8_model.compile(
        optimizer=optimizer,
        classification_loss="binary_crossentropy",
        box_loss="ciou",
    )
```

| Hiperparámetro | Valor |
| :--- | :--- |
| *Backbone* | `yolo_v8_xs_backbone` (`include_rescaling=True`) |
| Profundidad FPN | 5 |
| Formato de cajas | `xyxy` |
| Optimizador | Adam — `lr=0.0007`, `weight_decay=0.0009`, `global_clipnorm=10.0` |
| Pérdida de clasificación | `binary_crossentropy` |
| Pérdida de caja | `ciou` |
| *Batch size* | 16 |
| Resolución de entrada | 640×640 (`JitteredResize`, `scale_factor=(0.8, 1.25)`) |
| Estrategia de distribución | `tf.distribute.MirroredStrategy` |
| *Callbacks* | `ModelCheckpoint` (mejor `val_loss`), `ReduceLROnPlateau` (`factor=0.01`, `patience=8`), `EarlyStopping` (`patience=20`) |

El entorno de desarrollo y entrenamiento fue el PC de escritorio descrito en la Tabla 2 (Ryzen 7 5700X, 32 GB RAM, RTX 4060 Ti).

### 4.4 Adaptación propuesta: Head con capas convolucionales separables

Se reemplazaron las capas `Conv2D` estándar por capas `SeparableConv2D` **únicamente en el módulo Head**, dejando Backbone y Neck sin modificar. La sustitución descompone cada convolución en:

1. Una **convolución en profundidad (*depthwise*)**, que aplica un filtro por canal de entrada de forma independiente, capturando patrones espaciales específicos.
2. Una **convolución punto a punto (*pointwise*)**, que fusiona la información de los diferentes canales mediante filtros $1\times1$, aportando la combinación intercanal necesaria para la clasificación y la detección.

En términos funcionales, la sustitución impacta las **ramas de predicción del Head** y, a través de ellas, la integración de las características multiescala provenientes del Neck.

> **Nota de implementación:** la variante separable requiere una **compilación modificada de `keras_cv`** (`keras_cv-0.9.0.1-py3-none-any.whl`, instalada localmente en el notebook `02`), ya que la librería oficial no expone un Head basado en `SeparableConv2D`.

### 4.5 Criterio de selección del número de épocas

La selección del número de épocas se hizo mediante un **enfoque incremental**, evaluando el desempeño en intervalos de **10 épocas** y observando la evolución de métricas clave como mAP y precisión. Ambos modelos se entrenaron con el mismo conjunto de entrenamiento y se evaluaron sobre el mismo conjunto de validación de COCO2017, bajo métricas de desempeño idénticas, de modo que las diferencias observadas pudieran atribuirse principalmente a las características de las arquitecturas y no a variaciones en los datos o en las condiciones de evaluación.

### 4.6 Hardware de prueba

| Componente | PC Escritorio | Raspberry Pi 4 |
| :--- | :--- | :--- |
| **Procesador** | Ryzen 7 5700X (8C/16T, 3.4 GHz) | BCM2711 (ARM Cortex-A72, 1.5 GHz) |
| **GPU** | RTX 4060 Ti (8 GB GDDR6) | Sin GPU (CPU con NEON SIMD) |
| **RAM** | 32 GB DDR4 @ 3200 MHz | 8 GB LPDDR4 @ 2133 MHz |
| **Almacenamiento** | SSD NVMe 1 TB (PCIe 4.0) | microSD 128 GB |
| **Sistema operativo** | Windows 10 Pro 64-bit | Raspberry Pi OS 64-bit |
| **Eficiencia energética** | Alto consumo (65 W CPU, 160 W GPU) | Bajo consumo (~7 W) |
| **Procesamiento** | Soporte CUDA / TensorRT | Dependiente de CPU |

<p align="center"><em>Tabla 2: Especificaciones del hardware de prueba.</em></p>

### 4.7 Métricas de evaluación

#### Precision

Mide la proporción de predicciones correctas en relación con todas las predicciones realizadas:

$$\text{Precision} = \frac{TP}{TP + FP}$$

donde $TP$ (*True Positives*) es el número de detecciones correctas en las que el objeto detectado coincide con el objeto verdadero, y $FP$ (*False Positives*) el número de detecciones incorrectas en las que el modelo predice un objeto inexistente. Un valor alto indica pocas detecciones erróneas, algo fundamental donde los falsos positivos tienen consecuencias críticas.

#### Recall (sensibilidad)

Mide la capacidad del modelo para detectar **todos** los objetos presentes en una imagen:

$$\text{Recall} = \frac{TP}{TP + FN}$$

donde $FN$ (*False Negatives*) es el número de objetos reales que el modelo no detectó. Un recall alto minimiza omisiones, aunque puede venir acompañado de un mayor número de falsos positivos.

#### Intersection over Union (IoU)

Cuantifica el solapamiento entre la caja delimitadora predicha y la caja real (*ground truth*):

$$\text{IoU} = \frac{\text{Área}_{\text{Intersección}}}{\text{Área}_{\text{Unión}}}$$

con:

$$\text{Área}_{\text{Intersección}} = \max(0,\; x_{\text{derecha}} - x_{\text{izquierda}}) \times \max(0,\; y_{\text{inferior}} - y_{\text{superior}})$$

$$\text{Área}_{\text{Unión}} = \text{Área}_{\text{predicha}} + \text{Área}_{\text{real}} - \text{Área}_{\text{Intersección}}$$

donde $x_{\text{izquierda}}, y_{\text{superior}}$ son las coordenadas de la esquina superior izquierda de cada caja y $x_{\text{derecha}}, y_{\text{inferior}}$ las de la esquina inferior derecha.

Umbrales habituales:

| Umbral | Criterio | Uso típico |
| :--- | :--- | :--- |
| **IoU = 0,50** | Permisivo: basta un 50 % de solapamiento | Videovigilancia, control de tráfico |
| **IoU = 0,75** | Estricto: exige mayor precisión espacial | Inspección automatizada de productos |
| **IoU = 0,95** | Extremadamente exigente | Objetos pequeños o parcialmente ocultos; ensamblaje electrónico |

Además de evaluar, el IoU interviene en el entrenamiento: ajusta la regresión de cajas minimizando la función de pérdida, determina si una detección es TP o FP, y sostiene el proceso de NMS. Un umbral mal calibrado afecta significativamente el rendimiento reportado: uno demasiado alto vuelve el criterio de coincidencia tan exigente que muchas detecciones válidas no se contabilizan como aciertos; uno demasiado bajo relaja el criterio y acepta detecciones con menor exigencia de precisión espacial.

#### Average Precision (AP)

Área bajo la curva de precisión-recall, que resume el desempeño a través de distintos umbrales de confianza:

$$AP = \int_{0}^{1} P(r)\, dr$$

donde $P(r)$ representa la curva de precisión-recall interpolada.

#### Mean Average Precision (mAP)

Promedio de los valores de AP a través de todas las clases del conjunto:

$$mAP = \frac{1}{N} \sum_{i=1}^{N} AP_i$$

donde $N$ es el número total de clases y $AP_i$ la precisión promedio de la clase $i$.

- **mAP50** — promedio calculado con un umbral de IoU fijo del 50 %. Criterio permisivo; proporciona una referencia inicial de qué tan bien el modelo detecta objetos sin exigir la alineación exacta de las cajas. **Es la métrica principal reportada en este trabajo.**
- **mAP50-95** — promedio sobre 10 umbrales de IoU, desde 0,50 hasta 0,95 en intervalos de 0,05:

$$mAP_{50-95} = \frac{1}{10N} \sum_{j=1}^{10} \sum_{i=1}^{N} AP_{i,\, \text{IoU}=0{,}50 + 0{,}05(j-1)}$$

  donde $j$ representa cada uno de los 10 umbrales, $N$ el número total de clases y $AP_{i,\text{IoU}}$ la precisión promedio de la clase $i$ en un umbral específico. Es el estándar de la literatura moderna por reflejar el desempeño desde detecciones fáciles (IoU = 0,50) hasta detecciones altamente precisas y restrictivas (IoU = 0,95).

#### Métricas de eficiencia

| Métrica | Definición |
| :--- | :--- |
| **Velocidad de inferencia (FPS)** | Imágenes procesadas por segundo en cada entorno de ejecución. |
| **Tiempo por imagen ($\bar{t} \pm \sigma$)** | Latencia media y su desviación estándar; la dispersión mide la estabilidad cuadro a cuadro. |
| **Consumo de memoria / recursos** | Huella de parámetros y uso de recursos en cada dispositivo. |

---

## 5. DESARROLLO Y RESULTADOS

### 5.1 Comparación de parámetros del modelo

| Modelo | Parámetros totales | Parámetros entrenables | Parámetros no entrenables |
| :--- | ---: | ---: | ---: |
| **YOLOv8 Modificado (Separable)** | **1.258.715 (4,80 MB)** | 1.245.819 (4,75 MB) | 12.896 (50,38 KB) |
| **YOLOv8 Base (Conv2D)** | 3.991.584 (15,23 MB) | 3.978.688 (15,18 MB) | 12.896 (50,38 KB) |

<p align="center"><em>Tabla 3: Parámetros del modelo implementado de YOLOv8.</em></p>

La sustitución en el Head redujo los parámetros totales de ~3,99 millones a ~1,25 millones, una **reducción cercana al 68 %**, con la consiguiente caída en el consumo de memoria. La pregunta que aborda el resto de esta sección es si esa compactación se traduce en una ganancia de velocidad utilizable y cuánta calidad de detección cuesta.

### 5.2 Costo de entrenamiento

El entrenamiento fue la parte crítica del costo experimental de la tesis:

- Por la densidad del modelo, el tamaño de COCO2017 (118.287 imágenes de entrenamiento y 5.000 de validación) y la resolución de entrada (640×640), **cada época —incluyendo entrenamiento y evaluación sobre el conjunto de validación— tardó entre 10 y 12 horas** en la GPU utilizada.
- Entrenar 50 épocas del modelo Conv2D y 100 épocas del modelo Separable supuso **varias semanas de cómputo efectivo**.
- Esta limitación temporal condicionó el número de épocas ejecutables y, en consecuencia, **restringió la exploración sistemática de hiperparámetros** (tamaño de lote, tasa de aprendizaje, entre otros).
- El tiempo real del proyecto aumentó por contingencias como cortes de energía, errores de código y reinicios de entrenamiento.

### 5.3 Evolución del rendimiento: mAP por cada 10 épocas

![Figura 6: Evolución de la métrica mAP por cada 10 épocas](docs/images/figura6_map_epocas.png)

<p align="center"><em>Figura 6: Evolución de la métrica mAP por cada 10 épocas. (a) Modelo YOLOv8-Conv2D. (b) Modelo YOLOv8-Separable.</em></p>

- **(a) YOLOv8-Conv2D:** el mAP crece de forma monótona hasta ~0,21 en la época 50 y luego desciende, lo que sugiere el inicio de **sobreajuste (*overfitting*)**. Por eso se fijó la configuración final en **50 épocas**.
- **(b) YOLOv8-Separable:** la convergencia es **más lenta y notoriamente más ruidosa**, oscilando entre 0,10 y 0,18 sin una tendencia ascendente limpia. Requirió un mayor número de iteraciones para aproximarse a su mejor desempeño observado, estabilizándose en torno a las **100 épocas**.

### 5.4 Resultados de validación por modelo

#### YOLOv8-Conv2D (50 épocas)

| Métrica | Valor |
| :--- | :--- |
| Modelo | YOLOv8-Conv2D |
| Épocas | 50 |
| **mAP50** | **0,2119** |
| IoU promedio | 0,8367 |
| **Precisión** | **0,6836** |
| **Recall** | **0,3260** |
| FPS CPU | 2,34 |
| FPS GPU | 5,48 |
| FPS Raspberry Pi | 0,07 |

<p align="center"><em>Tabla 4: Resultados del modelo YOLOv8-Conv2D.</em></p>

![Figura 7: Modelo de detección con YOLOv8-Conv2D entrenado con 50 épocas](docs/images/figura7_deteccion_conv2d.png)

<p align="center"><em>Figura 7: Modelo de detección con YOLOv8-Conv2D entrenado con 50 épocas. En rojo las cajas reales (ground truth), en amarillo las predicciones del modelo sobre imágenes del conjunto de validación de COCO2017.</em></p>

La evidencia cualitativa muestra que Conv2D mantiene **detecciones más estables, especialmente con objetos medianos y pequeños**, y cajas delimitadoras más ajustadas.

#### YOLOv8-Separable (100 épocas)

| Métrica | Valor |
| :--- | :--- |
| Modelo | YOLOv8-Separable |
| Épocas | 100 |
| mAP50 | 0,1751 |
| IoU promedio | 0,8302 |
| Precisión | 0,6734 |
| Recall | 0,2933 |
| FPS CPU | 2,41 |
| FPS GPU | 5,36 |
| **FPS Raspberry Pi** | **0,68** |

<p align="center"><em>Tabla 5: Resultados del modelo YOLOv8-Separable.</em></p>

![Figura 8: Modelo de detección con YOLOv8-Separable entrenado con 100 épocas](docs/images/figura8_deteccion_separable.png)

<p align="center"><em>Figura 8: Modelo de detección con YOLOv8-Separable entrenado con 100 épocas. Mismo protocolo de visualización y mismas imágenes de validación que la Figura 7, para comparación directa.</em></p>

### 5.5 Tabla general de resultados de validación e inferencia

| Métrica / Dispositivo | YOLOv8-Separable (100 épocas) | YOLOv8-Conv2D (50 épocas) |
| :--- | :---: | :---: |
| **Épocas de entrenamiento** | 100 | 50 |
| **mAP50** | 0,1751 | **0,2119** |
| **IoU promedio** | 0,8302 | **0,8367** |
| **Precisión** | 0,6734 | **0,6836** |
| **Recall** | 0,2933 | **0,3260** |
| **FPS en CPU** | **2,41** | 2,34 |
| **FPS en GPU** | 5,36 | **5,48** |
| **FPS en Raspberry Pi** | **0,68** | 0,07 |
| **Tiempo promedio CPU ($\bar{t} \pm \sigma$) [s]** | **0,4156 ± 0,0136** | 0,4273 ± 0,0207 |
| **Tiempo promedio GPU ($\bar{t} \pm \sigma$) [s]** | 0,1866 ± 0,1671 | **0,1824 ± 0,0169** |
| **Tiempo promedio Raspberry Pi ($\bar{t} \pm \sigma$) [s]** | **1,4754 ± 0,0142** | 13,9322 ± 1,5230 |

<p align="center"><em>Tabla 6: Resultados en validación para ambos modelos.</em></p>

### 5.6 Interpretación de los resultados

#### Calidad de detección

**YOLOv8-Conv2D obtiene el mejor desempeño en todas las métricas de calidad** (mAP 0,2119 vs. 0,1751; y mejores IoU promedio, precisión y recall). Esto es coherente con su diseño: las capas convolucionales 2D estándar dan a la red una mayor capacidad de representación, lo que le permite modelar patrones más complejos en las imágenes de COCO2017 y se traduce en mejores cajas delimitadoras y una mayor tasa de verdaderos positivos, a costa de un incremento significativo en el número de parámetros y en el costo de cómputo asociado.

Un punto clave del experimento: **la variante separable fue entrenada por el doble de épocas (100 vs. 50) y aun así no supera a Conv2D**. Esto indica que la diferencia observada se explica principalmente por **diferencias estructurales de capacidad representacional**, no por falta de convergencia.

#### Velocidad en hardware acelerado

En el computador de escritorio, ambos modelos presentan valores de FPS **muy similares tanto en CPU como en GPU** (2,34–2,41 FPS en CPU; 5,36–5,48 FPS en GPU). Esto sugiere que en ese escenario el cuello de botella está principalmente en:

- la carga de datos,
- las transferencias memoria↔GPU,
- y el propio entorno de ejecución de TensorFlow,

más que en el número exacto de parámetros del modelo. Una GPU moderna (RTX 4060 Ti) **amortigua** el aumento de complejidad de Conv2D, de modo que la ventaja teórica en FLOPs de la versión separable no se traduce en una ganancia de velocidad marcada.

#### Velocidad en hardware restringido

La situación **cambia de forma notable en la Raspberry Pi**, donde el procesamiento depende exclusivamente de la CPU ARM Cortex-A72 y no existe aceleración por GPU:

| | YOLOv8-Conv2D | YOLOv8-Separable | Mejora |
| :--- | ---: | ---: | ---: |
| **FPS** | 0,07 | 0,68 | **≈ 9,7×** |
| **Tiempo por imagen** | 13,9322 s | 1,4754 s | **≈ 9,4× más rápido** |
| **Desviación estándar** | ± 1,5230 s | ± 0,0142 s | **≈ 107× menos dispersión** |

Es decir, la versión separable no solo es casi **un orden de magnitud más rápida**, sino también **mucho más estable en la latencia cuadro a cuadro** — un requisito crítico en aplicaciones de tiempo real sobre hardware embebido.

#### Cuellos de botella en la exportación a TensorFlow Lite

Las diferencias de estabilidad están directamente vinculadas a **cómo fue necesario adaptar los modelos a TensorFlow Lite** para ejecutarlos en la Raspberry Pi:

1. **Operaciones no soportadas.** Durante la exportación se detectó que ciertas operaciones —en particular **`DepthwiseConv2dNative`**— no estaban soportadas por los núcleos nativos de TFLite.
2. **Habilitación de Select TF Ops.** Fue necesario activar **Select TF Ops** (`tf.lite.OpsSet.SELECT_TF_OPS`) y **reestructurar el grafo del modelo** para lograr la conversión.
3. **Separación del grafo de inferencia.** Para una ejecución correcta hubo que **separar explícitamente la inferencia pura** de las etapas asociadas al codificador de etiquetas y al proceso de **Non-Maximum Suppression (NMS)**, manteniendo en el modelo convertido únicamente las capas imprescindibles para la predicción.
4. **Costo residual de las operaciones flex.** Esa reestructuración permitió desplegar la arquitectura entrenada, pero implicó que **una fracción del grafo se ejecutara como operaciones *flex***, que no siempre pueden beneficiarse plenamente de las optimizaciones internas de TFLite ni de la librería **XNNPACK**.

El impacto de esta limitación es **más evidente en el modelo Conv2D**, más pesado y con mayor ancho de banda de memoria, lo que explica tanto sus tiempos de inferencia más altos como la mayor variabilidad observada en las mediciones (± 1,52 s frente a ± 0,014 s).

### 5.7 Visualizaciones generadas desde los resultados

Los gráficos siguientes se construyen directamente a partir de las tablas anteriores mediante [`outputs/make_figures.py`](outputs/make_figures.py); ningún valor fue alterado ni estimado.

| | |
| :---: | :---: |
| ![FPS por dispositivo](outputs/figures/fps_by_device.png) | ![Latencia por dispositivo](outputs/figures/latency_by_device.png) |
| ![Comparación de tamaño de modelo](outputs/figures/model_size_comparison.png) | ![Trade-off mAP vs latencia](outputs/figures/map_vs_latency_tradeoff.png) |

**Gráfico interactivo:** [`outputs/make_interactive.py`](outputs/make_interactive.py) genera además un HTML autocontenido (Plotly) con el trade-off precisión vs. latencia por dispositivo; al pasar el cursor sobre cada punto muestra FPS, latencia, mAP50 y número de parámetros para esa combinación de dispositivo y arquitectura. Se construye desde la misma tabla de mediciones y no se versiona en el repositorio: ejecutar el script para regenerarlo en `outputs/interactive/`.

**Convergencia real del entrenamiento (YOLOv8-Separable, log de 10 épocas):**

![Convergencia de entrenamiento — YOLOv8-Separable](outputs/figures/training_convergence_separable.png)

Esta curva no es un resumen estadístico: es la secuencia época a época de `box_loss` / `class_loss` / `val_loss` guardada en la celda de salida del propio notebook `02_yolov8_separable_conv.ipynb` (`Epoch 1/10 … Epoch 10/10`), incluida para que las métricas reportadas puedan auditarse contra el log de entrenamiento y no solo contra los números finales.

---

## 6. CONCLUSIONES Y FUTURAS LÍNEAS DE INVESTIGACIÓN

### 6.1 Conclusiones

**Sobre el objetivo comparativo.** Los resultados de validación demostraron que **YOLOv8-Conv2D presenta una ventaja en calidad de detección** sobre COCO2017 (mAP 0,2119 vs. 0,1751) **a pesar de haber sido entrenado con la mitad de épocas**, lo que refuerza que su mayor capacidad representacional impacta directamente en la precisión y la localización.

**Sobre el despliegue embebido.** En Raspberry Pi se evidencia el efecto esperado de la sustitución de Conv2D por Separable: la variante separable **reduce drásticamente la latencia y estabiliza la inferencia** (0,68 vs. 0,07 FPS; 1,48 s vs. 13,93 s por imagen), confirmando la relación entre eficiencia de cómputo y viabilidad de despliegue en entornos embebidos.

**Sobre el *trade-off*.** El modelo con convoluciones estándar ofrece mayor precisión pero con un costo de cálculo considerablemente más alto, reflejado en mayores tiempos de predicción y un uso más intensivo de procesamiento y memoria — lo que lo hace **menos viable para dispositivos embebidos**. El modelo con convoluciones separables mostró una **ligera reducción de precisión** (−3,7 puntos de mAP50, −0,65 puntos de IoU promedio, −1,0 punto de precisión) pero mantuvo un desempeño **suficientemente robusto para quedar dentro del umbral aceptable** en aplicaciones de visión por computadora, con una reducción del **68 % de parámetros**.

**Sobre la hipótesis.** La hipótesis planteada no se sostiene en su lectura literal —los modelos adaptados a hardware reducido **no** alcanzaron mayor precisión que los tradicionales— pero sí en su lectura **en función de los recursos disponibles**: sobre una plataforma sin aceleración dedicada, la variante separable entrega casi 10× más rendimiento con una pérdida marginal de calidad, lo que la convierte en la única de las dos configuraciones operativamente utilizable en ese contexto.

**Sobre el criterio de selección.** La elección de la arquitectura debe alinearse con los requisitos específicos de la aplicación:

| Contexto de despliegue | Modelo recomendado | Razón |
| :--- | :--- | :--- |
| Infraestructura con GPU/CPU de alto rendimiento y prioridad en calidad | **YOLOv8-Conv2D** | Mejor mAP, IoU, precisión y recall; la penalización de latencia es marginal cuando hay aceleración. |
| Sistemas embebidos / bajo consumo, con latencia estable como requisito operacional | **YOLOv8-Separable** | ≈ 9,7× más FPS y ≈ 107× menos dispersión de latencia en Raspberry Pi; huella de memoria 68 % menor. |

En conclusión, **la selección adecuada de la arquitectura del modelo de detección puede mejorar significativamente la eficiencia de procesamiento sin comprometer en exceso la precisión**, y la variante separable resulta una alternativa viable para sistemas anticolisión y otras aplicaciones de visión que deban operar de forma continua en dispositivos de bajo consumo.

### 6.2 Limitaciones y alcances de la generalización

- **Dominio:** la estimación del desempeño se restringió a **COCO2017**; la magnitud de las diferencias podría variar en dominios con distribuciones distintas.
- **Plataformas:** se evaluaron **dos** entornos (escritorio y Raspberry Pi 4); el comportamiento en hardware embebido *con* aceleración dedicada queda sin medir.
- **Asimetría experimental:** las configuraciones difirieron en número de épocas (50 para Conv2D, 100 para Separable). No obstante, la brecha persistente —a favor del modelo entrenado *menos* tiempo— sugiere que el factor dominante es arquitectónico y no de convergencia.
- **Flujo de conversión:** las mediciones en Raspberry Pi dependieron del proceso de conversión y despliegue en TensorFlow Lite, lo que puede afectar las latencias absolutas. **Los tiempos deben interpretarse principalmente de forma relativa entre modelos**, no como valores absolutos transferibles a otro flujo de exportación.
- **Presupuesto de cómputo:** el costo de 10–12 horas por época impidió una exploración sistemática de hiperparámetros; los valores de mAP alcanzados están por debajo de los reportados por implementaciones oficiales entrenadas con presupuestos mucho mayores.

### 6.3 Futuras líneas de investigación

**Evaluación del rendimiento en diferentes hardware**

| Plataforma | Qué permitiría evaluar |
| :--- | :--- |
| **Google Edge TPU** | Eficiencia de modelos optimizados en dispositivos IoT, con bajo consumo energético. |
| **NVIDIA Jetson** | Entorno estándar en robótica y automatización, donde la optimización de la latencia de inferencia es crítica. |
| **FPGAs y NPUs** | Configuraciones específicas para acelerar la inferencia de modelos de visión, con potenciales mejoras en eficiencia y consumo. |

**Otras direcciones**

- **Técnicas de compresión:** cuantización de pesos (INT8) y poda de parámetros (*pruning*), combinadas con aceleración por hardware, para mejorar la eficiencia sin sacrificar drásticamente el rendimiento.
- **Comparación con detectores ultraligeros:** NanoDet, MobileNet-SSD y PP-YOLO, para situar la variante separable dentro del estado del arte de modelos diseñados desde cero para el borde.
- **Recuperación de precisión:** *fine-tuning* específico de la variante separable para cerrar parte de la brecha de mAP.
- **Validación en despliegue real:** medición de latencia de detección, consumo energético y estabilidad en producción, sobre distintos conjuntos de datos y escenarios de aplicación.

---

## 7. ESTRUCTURA DEL REPOSITORIO

```
yolov8-separable-convolutions/
├── 01_yolov8_standard_conv2d.ipynb            # Entrenamiento y evaluación de YOLOv8-Conv2D (baseline)
├── 02_yolov8_separable_conv.ipynb             # Entrenamiento y evaluación de YOLOv8-Separable (Head separable)
├── docs/
│   └── images/
│       ├── figura1_ejemplo_yolo.png           # Fig. 1 — Detección de objetos en el mapa de tareas de visión
│       ├── figura2_arquitectura_yolov8.png    # Fig. 2 — Arquitectura Backbone / Neck / Head
│       ├── figura3_conv2d_vs_separable.png    # Fig. 3 — Conv2D vs SeparableConv2D
│       ├── figura4_procesos_convolucion.png   # Fig. 4 — Procesos de convolución estándar y separable
│       ├── figura5_proceso_entrenamiento.png  # Fig. 5 — Diseño experimental
│       ├── figura6_map_epocas.png             # Fig. 6 — mAP por cada 10 épocas
│       ├── figura7_deteccion_conv2d.png       # Fig. 7 — Detecciones YOLOv8-Conv2D (50 épocas)
│       └── figura8_deteccion_separable.png    # Fig. 8 — Detecciones YOLOv8-Separable (100 épocas)
├── outputs/
│   ├── make_figures.py                        # Genera los gráficos estáticos desde la tabla de resultados
│   ├── make_interactive.py                    # Genera el gráfico interactivo de trade-off (Plotly)
│   └── figures/
│       ├── fps_by_device.png                  # Generado — FPS por dispositivo
│       ├── latency_by_device.png              # Generado — latencia de inferencia por dispositivo
│       ├── map_vs_latency_tradeoff.png        # Generado — compromiso precisión vs. latencia
│       ├── model_size_comparison.png          # Generado — tamaño de los modelos en disco
│       ├── training_convergence_separable.png # Generado — convergencia del modelo separable
│       ├── CONV2_50Epoch.png                  # Original de tesis — inferencia YOLOv8-Conv2D
│       ├── SEP_100EPOCH.png                   # Original de tesis — inferencia YOLOv8-Separable
│       ├── mAP.png                            # Original de tesis — curvas mAP50
│       ├── comparacion modelos.png            # Original de tesis — Tabla 2.1, estado del arte
│       ├── comparcion.png                     # Original de tesis — Tabla 3, parámetros
│       ├── tabla 3.png                        # Original de tesis — resumen paramétrico
│       ├── tabla resultados tesis.png         # Original de tesis — Tabla 6 consolidada
│       └── yolo_arq.pdf                       # Original de tesis — esquema arquitectónico
├── .gitattributes                             # Normalización de fin de línea (LF)
├── README.md                                  # Este documento (español)
├── README.en.md                               # Versión en inglés
└── LICENSE                                    # MIT
```

> Las figuras bajo `docs/images/` son las que numera este documento (Figuras 1–8). Las de `outputs/figures/` son de dos tipos: las generadas por `make_figures.py` a partir de la tabla de resultados, y los recortes originales del documento de tesis subidos al repositorio, que en varios casos duplican contenido ya presente como tabla nativa o como figura numerada.

### Notebooks

| Notebook | Contenido |
| :--- | :--- |
| [`01_yolov8_standard_conv2d.ipynb`](01_yolov8_standard_conv2d.ipynb) | Modelo de control. Carga y transformación de anotaciones COCO, generadores `tf.data` con `JitteredResize`, construcción del `YOLOV8Detector` con Head estándar, entrenamiento, visualización de detecciones y cálculo propio de IoU / AP / mAP. |
| [`02_yolov8_separable_conv.ipynb`](02_yolov8_separable_conv.ipynb) | Modelo tratamiento. Mismo flujo, instalando la compilación modificada `keras_cv-0.9.0.1` con Head basado en `SeparableConv2D`. Conserva en sus celdas de salida el `model.summary()` con `Total params: 1,258,715 (4.80 MB)` y el log de entrenamiento. |

---

## 8. REPRODUCCIÓN DEL EXPERIMENTO

### Requisitos

```bash
python -m pip install tensorflow==2.13.1
python -m pip install keras_cv                             # notebook 01 (Head estándar)
python -m pip install keras_cv-0.9.0.1-py3-none-any.whl    # notebook 02 (Head separable, build modificado)
python -m pip install pandas tqdm scikit-learn matplotlib opencv-python
```

### Datos

Descargar COCO2017 y disponerlo con la siguiente estructura:

```
coco2017/
├── train2017/                                  # 118.287 imágenes
├── val2017/                                    #   5.000 imágenes
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Regenerar los gráficos

```bash
python outputs/make_figures.py
python outputs/make_interactive.py
```

### Nota de reproducibilidad

Los *benchmarks* de GPU / CPU / Raspberry Pi reportados en este README son las **mediciones originales de la tesis**, obtenidas sobre el hardware descrito en la Tabla 2 (escritorio Ryzen 7 5700X + RTX 4060 Ti, y una Raspberry Pi 4 física), con un entorno fijado en `tensorflow==2.13.1` y la compilación modificada de `keras_cv`.

Reproducirlos exactamente requiere esa misma combinación de hardware y software, el conjunto COCO2017 completo (~19 GB) y los *checkpoints* entrenados. Ejecutarlos en hardware distinto no "verificaría" los números de la tesis: produciría un *benchmark* diferente y no comparable. Por esa razón **ningún número de la Sección 5 fue recalculado ni sustituido** al redactar esta documentación; los únicos artefactos derivados son los gráficos de la Sección 5.7, generados directamente a partir de esas mediciones.

---

## 9. REFERENCIAS

Referencias principales citadas en el marco teórico y la metodología:

- Chollet, F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions*.
- Howard, A. G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*.
- Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition*.
- Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*.
- Lin, T.-Y., et al. (2014). *Microsoft COCO: Common Objects in Context*.
- Redmon, J., et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*.
- Bochkovskiy, A., et al. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection*.
- Wang, C.-Y., et al. (2019). *CSPNet: A New Backbone that can Enhance Learning Capability of CNN*.
- Terven, J., et al. (2023). *A Comprehensive Review of YOLO Architectures in Computer Vision*.
- Varghese, R., & M., S. (2024). *YOLOv8: A Novel Object Detection Algorithm with Enhanced Performance and Robustness*.
- Neubeck, A., & Van Gool, L. (2006). *Efficient Non-Maximum Suppression*.
- Han, S., et al. (2016). *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding*.
- Jouppi, N. P., et al. (2017). *In-Datacenter Performance Analysis of a Tensor Processing Unit*.
- Liu, L., et al. (2020). *Deep Learning for Generic Object Detection: A Survey*.
- Zou, Z., et al. (2019). *Object Detection in 20 Years: A Survey*.
- Ultralytics (2024). *YOLOv8 Documentation*.

> Las Figuras 1 a 8 provienen del documento de tesis y de la presentación de defensa. La Figura 2 reproduce el diagrama arquitectónico de Terven et al. (2023).

---

## LICENCIA

MIT — ver [LICENSE](LICENSE).

---

## AUTOR

**Pablo Vicente Reyes Pino**
Proyecto de Tesis — Ingeniería Civil en Computación e Informática
Universidad Mayor · Santiago, Chile · Abril 2026
Profesor Guía: Dr. Anthony D. Cho · Profesor Revisor: Carlos Muñoz
