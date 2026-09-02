[ 🇺🇸 [Read in English](README.md) ] | [ 🇨🇱 Español ]

# Evaluación de la Eficiencia de YOLOv8 en GPU, CPU y Raspberry Pi: Convoluciones Estándar vs. Separables

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-CUDA%20%7C%20TensorRT-FF6F00?logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-XNNPACK-FF6F00?logo=tensorflow&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi%204-despliegue%20edge-A22846?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> **Proyecto de Tesis:** Evaluación del compromiso entre desempeño en detección de objetos y eficiencia computacional del modelo YOLOv8.

---

## 📌 Descripción General

Este repositorio contiene la estructura, metodología y resultados experimentales desarrollados en la tesis *"Evaluación de la eficiencia de YOLOv8 en GPU, CPU y Raspberry Pi: Convoluciones estándar vs separables"*.

El objetivo central de la investigación es analizar y evaluar la arquitectura de detección de objetos **YOLOv8** implementada en entornos con restricciones computacionales. Se realiza una comparación entre la versión base con convoluciones 2D tradicionales (**YOLOv8-Conv2D**) y una variante adaptada mediante el uso de convoluciones separables en profundidad (**YOLOv8-Separable**) en el módulo Head.

---

## 🧭 Planteamiento del Problema

Los modelos de detección de objetos como YOLOv8 casi siempre se diseñan y evalúan sobre GPUs de escritorio, donde las convoluciones 2D estándar resultan baratas gracias a los núcleos tensoriales dedicados. En la práctica, sin embargo, gran parte de los despliegues reales — cámaras de tráfico, sensores agrícolas, líneas de inspección industrial, robótica móvil — no cuentan con GPU disponible en el momento de la inferencia. Se ejecutan en CPU, o en placas embebidas como una Raspberry Pi, donde cada operación de multiplicación-acumulación se paga directamente en latencia y consumo energético.

Una convolución 2D estándar con tamaño de kernel `k`, `Cin` canales de entrada y `Cout` canales de salida cuesta `k² · Cin · Cout` operaciones de multiplicación-acumulación por píxel de salida. Una **convolución separable en profundidad** factoriza esa operación en un paso *depthwise* (un filtro por canal de entrada, costo `k² · Cin`) seguido de una convolución puntual 1×1 que mezcla canales (costo `Cin · Cout`). El costo combinado, `k² · Cin + Cin · Cout`, crece de forma aditiva en vez de multiplicativa respecto al tamaño del kernel — para un kernel 3×3 típico esto elimina aproximadamente un orden de magnitud de aritmética, a costa de un espacio de funciones más reducido y, por lo tanto, en general, de menor capacidad representacional por capa.

Este es exactamente el compromiso que esta tesis busca **medir, no asumir**: **¿reemplazar las convoluciones estándar por convoluciones separables en profundidad en el Head de YOLOv8 se traduce realmente en una ganancia de velocidad utilizable en hardware limitado por CPU o embebido, y cuánta calidad de detección (mAP, precisión, recall) se sacrifica a cambio?** La pregunta importa porque la reducción teórica de FLOPs no produce automáticamente una reducción proporcional de latencia — los límites de ancho de banda de memoria, el overhead del framework y las implementaciones de kernels específicas de cada hardware (p. ej. XNNPACK en ARM, TensorRT en GPU) interactúan con el número de operadores de formas que solo un benchmark empírico puede revelar. Ambas arquitecturas se entrenaron desde cero sobre COCO2017 y se evaluaron de extremo a extremo (mismos pesos, mismo preprocesamiento, mismo protocolo de evaluación) en tres perfiles de hardware genuinamente distintos — una GPU de escritorio, una CPU de escritorio y una Raspberry Pi 4 — de modo que la comparación aísla el efecto del tipo de convolución y no una variable de confusión proveniente del entrenamiento o la evaluación.

---

## 🎯 Objetivos del Proyecto

* **Analizar la arquitectura:** Estudiar los componentes base de YOLOv8 (Backbone, Neck y Head) y sus optimizaciones.
* **Desarrollar variantes:** Implementar la arquitectura clásica de YOLOv8 y la adaptación modificada con capas de convolución separable en profundidad.
* **Evaluación experimental:** Comparar el desempeño en métricas de precisión, recall, IoU, mAP, velocidad de inferencia y consumo de recursos.
* **Despliegue en hardware reducido:** Determinar la viabilidad operativa en plataformas embebidas (Raspberry Pi 4) frente a computadores de escritorio (GPU/CPU).

---

## 🛠️ Técnicas Utilizadas

| Técnica | Rol en este proyecto |
| :--- | :--- |
| **Arquitectura YOLOv8 (Backbone / Neck / Head)** | Arquitectura base de detección de objetos (preset `yolo_v8_xs_backbone` de `keras_cv`), utilizada como línea base común para ambas variantes. |
| **Convoluciones 2D estándar (YOLOv8-Conv2D)** | Arquitectura de control — el módulo Head usa capas `Conv2D` ordinarias, `k² · Cin · Cout` MACs/píxel. |
| **Convoluciones separables en profundidad (YOLOv8-Separable)** | Arquitectura de tratamiento — las convoluciones del módulo Head se factorizan en capas depthwise + pointwise (build parchado de `keras_cv`), reduciendo el número de parámetros y el costo aritmético. |
| **TensorFlow / Keras + CUDA / TensorRT** | Runtime de entrenamiento e inferencia en GPU sobre el PC de escritorio (entrenamiento con estrategia distribuida, checkpoints `.h5`). |
| **TensorFlow Lite + XNNPACK** | Ruta de exportación/ejecución usada para correr la inferencia en la Raspberry Pi 4 (solo CPU, sin acelerador dedicado). |
| **Protocolo de evaluación COCO2017** | Entrenamiento sobre el split de entrenamiento (118.287 imágenes), evaluación sobre el split de validación reservado (5.000 imágenes), a resolución de entrada 640×640, 80 clases de objetos. |
| **Métricas de evaluación** | mAP50, IoU promedio, precisión, recall, tamaño del modelo (MB) / número de parámetros, FPS y latencia por imagen (media ± desviación estándar) — el conjunto estándar de métricas para comparar calidad de detección contra costo de despliegue. |

---

## 🛠️ Entornos de Hardware Evaluados

| Especificación | PC de Escritorio | Raspberry Pi 4 |
| :--- | :--- | :--- |
| **Procesador** | AMD Ryzen 7 5700X (8C/16T, 3.4 GHz) | Broadcom BCM2711 (ARM Cortex-A72, 1.5 GHz) |
| **GPU** | NVIDIA RTX 4060 Ti (8 GB GDDR6) | Sin GPU dedicada (Procesamiento por CPU) |
| **Memoria RAM** | 32 GB DDR4 | 8 GB LPDDR4 |
| **Framework / Ejecución** | TensorFlow / Keras (CUDA / TensorRT) | TensorFlow Lite / XNNPACK |

---

## 📊 Dataset Utilizado

* **Base de datos:** COCO2017.
* **Entrenamiento:** 118,287 imágenes.
* **Validación:** 5,000 imágenes.
* **Categorías:** 80 clases de objetos etiquetados.
* **Resolución de entrada:** **640×640** píxeles.

---

## 📈 Resultados Resumidos

> Todos los números de esta sección son las **mediciones originales producidas durante el trabajo de tesis** (entrenamientos y benchmarks de hardware ejecutados en el PC y en la Raspberry Pi 4 descritos arriba). Provienen directamente de las tablas de resultados de la tesis y de las salidas guardadas en las celdas de `20250131 - YOLOv8_Separable.ipynb` (p. ej. la salida de `model.summary()` que reporta `Total params: 1,258,715 (4.80 MB)`, y el registro de entrenamiento de 10 épocas). Ningún número de esta sección fue recalculado ni re-benchmarkeado en la máquina usada para redactar esta documentación — ver la [Nota de reproducibilidad](#-nota-de-reproducibilidad) para el detalle de por qué, y qué se verificó realmente aquí.

### 1. Parámetros y Métricas de Calidad de Detección

| Métrica / Parámetro | YOLOv8-Conv2D | YOLOv8-Separable |
| :--- | :--- | :--- |
| **Parámetros Totales** | 3,991,584 (15.23 MB) | **1,258,715 (4.80 MB)** |
| **Épocas de Entrenamiento** | 50 épocas | 100 épocas |
| **mAP50** | **0.2119** | 0.1751 |
| **IoU Promedio** | **0.8367** | 0.8302 |
| **Precisión** | **0.6836** | 0.6734 |
| **Sensibilidad (Recall)** | **0.3260** | 0.2933 |

### 2. Rendimiento y Velocidad de Inferencia

| Entorno de Prueba | Modelo | FPS | Tiempo Promedio por Imagen ($\overline{t} \pm \sigma$) |
| :--- | :--- | :--- | :--- |
| **PC - GPU (RTX 4060 Ti)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | **5.48 FPS**<br>5.36 FPS | $0.1824 \pm 0.0169\text{ s}$<br>$0.1866 \pm 0.1671\text{ s}$ |
| **PC - CPU (Ryzen 7)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 2.34 FPS<br>**2.41 FPS** | $0.4273 \pm 0.0207\text{ s}$<br>$0.4156 \pm 0.0136\text{ s}$ |
| **Raspberry Pi 4** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 0.07 FPS<br>**0.68 FPS** | $13.9322 \pm 1.5230\text{ s}$<br>**$1.4754 \pm 0.0142\text{ s}$** |

### 3. Convergencia real de entrenamiento (YOLOv8-Separable, registro de 10 épocas)

El gráfico siguiente no es una estadística resumen — es la secuencia real, época a época, de `box_loss` / `class_loss` / `val_loss` guardada en la propia celda de salida del notebook para el entrenamiento de YOLOv8-Separable (`Epoch 1/10 … Epoch 10/10`), incluida para que las métricas reportadas puedan auditarse contra el registro de entrenamiento crudo y no solo contra los valores finales.

![Convergencia de entrenamiento — YOLOv8-Separable](outputs/figures/training_convergence_separable.png)

---

## 📊 Visualizaciones

Todos los gráficos siguientes se construyen directamente a partir de las tablas anteriores — ningún número fue alterado ni estimado para los gráficos.

| | |
| :---: | :---: |
| ![FPS por entorno](outputs/figures/fps_by_device.png) | ![Latencia por entorno](outputs/figures/latency_by_device.png) |
| ![Comparación de tamaño de modelo](outputs/figures/model_size_comparison.png) | ![Compromiso mAP vs. latencia](outputs/figures/map_vs_latency_tradeoff.png) |

**Gráfico interactivo:** [Compromiso Precisión vs. Latencia por Entorno](https://htmlpreview.github.io/?https://github.com/Rxyxs/yolov8-separable-convolutions/blob/main/outputs/interactive/latency_map_tradeoff.html) — pasa el cursor sobre cualquier punto para ver FPS, latencia, mAP50 y número de parámetros para esa combinación entorno/arquitectura (HTML autocontenido, construido con Plotly a partir de la misma tabla de benchmark real).

---

## 💡 Conclusiones Clave

* **Desempeño en Infraestructura con Aceleración:** La versión con convoluciones estándar (**YOLOv8-Conv2D**) logra una calidad de detección superior (mayor mAP y Recall). En equipos de escritorio con GPU o CPU potente, las diferencias en tiempo de inferencia son marginales.
* **Optimización de Parámetros:** El reemplazo por convoluciones separables en profundidad (**YOLOv8-Separable**) reduce el número total de parámetros en aproximadamente un **68%**.
* **Impacto en Dispositivos Embebidos:** En entornos sin aceleración dedicada como la Raspberry Pi, **YOLOv8-Separable** incrementa la velocidad de ejecución en casi un orden de magnitud (de 0.07 a 0.68 FPS) y garantiza mayor estabilidad en la latencia cuadro a cuadro.

---

## 🔁 Nota de Reproducibilidad

Los benchmarks de GPU/CPU/Raspberry Pi anteriores se produjeron en el hardware específico descrito en esta tesis (PC de escritorio con Ryzen 7 5700X + RTX 4060 Ti, y una Raspberry Pi 4 física), usando un entorno fijado a `tensorflow==2.13.1` y un build parchado de `keras_cv` con el Head de convoluciones separables. Reproducirlos exactamente requiere esa misma combinación de hardware/software más el dataset COCO2017 completo (~19 GB) y los checkpoints de los modelos entrenados, nada de lo cual es práctico de re-ejecutar desde una máquina arbitraria usada solo para editar documentación — y volver a correrlos en hardware distinto no "verificaría" los números de la tesis, sino que simplemente produciría un benchmark diferente y no comparable. Por esa razón, esta actualización de documentación no intenta reproducir ni reemplazar las cifras de GPU/CPU/Raspberry Pi: cada número en la sección de Resultados es la medición original de la tesis, y los únicos artefactos nuevos agregados aquí son los gráficos y el gráfico interactivo, generados directamente a partir de esos números existentes (`outputs/make_figures.py`, `outputs/make_interactive.py`).

---

## Licencia

MIT — ver [LICENSE](LICENSE).

## 📜 Créditos y Referencia Académica

* **Autor:** Pablo Vicente Reyes Pino
* **Tutor:** Dr. Anthony D. Cho
* **Institución:** Universidad Mayor — Escuela de Ingeniería Civil en Computación e Informática
* **Ubicación y Fecha:** Santiago, Chile — Abril 2026
