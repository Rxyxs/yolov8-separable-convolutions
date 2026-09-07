# Evaluación de la Eficiencia de YOLOv8 en GPU, CPU y Raspberry Pi: Convoluciones Estándar vs. Separables

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-CUDA%20%7C%20TensorRT-FF6F00?logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-XNNPACK-FF6F00?logo=tensorflow&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi%204-Edge%20Deployment-A22846?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Proyecto de Tesis:** Evaluación del compromiso (*trade-off*) entre calidad de detección de objetos y eficiencia computacional en arquitecturas YOLOv8 en entornos restringidos.

---

## 📌 Datos Académicos

* **Autor:** Pablo Vicente Reyes Pino
* **Profesor Guía:** Dr. Anthony D. Cho
* **Profesor Revisor:** Carlos Muñoz
* **Institución:** Escuela de Ingeniería Civil en Computación e Informática, Facultad de Ciencias, Ingeniería y Tecnología, Universidad Mayor (Santiago, Chile)
* **Fecha:** Abril / Julio 2026

---

## 📌 Resumen y Motivación

La detección de objetos en tiempo real es fundamental en sistemas de visión artificial modernos, como conducción autónoma, videovigilancia y robótica industrial. No obstante, los modelos de aprendizaje profundo tradicionales requieren una elevada capacidad de procesamiento que dificulta su despliegue en dispositivos de borde o hardware reducido.

Esta investigación evalúa el comportamiento del modelo **YOLOv8** modificando su módulo de detección (*Head*) mediante la sustitución de convoluciones 2D estándar (**YOLOv8-Conv2D**) por convoluciones separables en profundidad (**YOLOv8-Separable**). Ambos modelos fueron probados bajo un estricto protocolo experimental idéntico a lo largo de tres perfiles de hardware distintos: GPU dedicada, CPU de escritorio y una plataforma embebida Raspberry Pi 4.

---

## 🔬 Hipótesis de Investigación

> *"Los modelos de Deep Learning adaptados para hardware reducido tendrán una precisión mayor que los modelos de Deep Learning tradicionales en función de los recursos disponibles, comparando métricas específicas de rendimiento."*

---

## 🧭 Marco Teórico y Modificación Arquitectónica

El modelo YOLOv8 organiza su flujo en tres módulos principales: **Backbone** (extracción de características con CSPDarknet y activación SiLU), **Neck** (fusión multiescala vía FPN y PANet con bloques C2f y SPPF) y **Head** (generación de predicciones de cajas y clases).

### Convolución Conv2D (Estándar)
Aplica un filtro tridimensional completo sobre los canales de entrada. El número de parámetros requeridos para un kernel de tamaño $k$ con $C_{\text{in}}$ canales de entrada y $C_{\text{out}}$ filtros de salida es:

$$P_{\text{conv\_std}} = k \times k \times C_{\text{in}} \times C_{\text{out}}$$

### Convolución Separable en Profundidad
Factoriza la operación convolucional en dos etapas independientes:
1. **Convolución Depthwise:** Aplica un filtro espacial $k \times k$ a cada canal de entrada de forma individual.
   $$P_{\text{depthwise}} = k \times k \times C_{\text{in}}$$
2. **Convolución Pointwise:** Realiza una proyección lineal $1 \times 1$ para mezclar la información entre canales.
   $$P_{\text{pointwise}} = C_{\text{in}} \times C_{\text{out}}$$

Esta sustitución estratégica en el módulo *Head* reduce la complejidad computacional y el número de parámetros del modelo en aproximadamente un **68%**.

---

## 🛠️ Entornos de Hardware y Justificación de Plataformas

| Especificación | PC de Escritorio (GPU) | PC de Escritorio (CPU) | Raspberry Pi 4 | ESP32-CAM |
| :--- | :--- | :--- | :--- | :--- |
| **Procesador** | AMD Ryzen 7 5700X (8C/16T, 3.4 GHz) | AMD Ryzen 7 5700X | Broadcom BCM2711 (Quad-core ARM Cortex-A72 @ 1.5 GHz) | ESP32-D0WDQ6 Dual-core @ 240 MHz |
| **Acelerador Gráfico** | NVIDIA RTX 4060 Ti (8 GB GDDR6) | Sin GPU dedicada | Sin GPU dedicada (ARM NEON SIMD) | Sin acelerador |
| **Memoria RAM** | 32 GB DDR4 3200 MHz | 32 GB DDR4 | 8 GB LPDDR4 2133 MHz | 520 KB SRAM + 4 MB PSRAM |
| **Consumo Energético** | ~160W (GPU) + 65W (CPU) | ~65W | **~7W** | Ultra bajo |
| **Runtime / Framework** | TensorFlow / Keras (CUDA / TensorRT) | TensorFlow CPU | TensorFlow Lite / XNNPACK | Incompatible |
| **Estado de Despliegue** | Evaluado | Evaluado | **Evaluado** | **Descartado** (memoria insuficiente para cargar el grafo de YOLOv8) |

---

## 📊 Configuración del Dataset y Entrenamiento

* **Base de datos:** COCO2017 (80 categorías de objetos etiquetados).
* **División de datos:** 118,287 imágenes para entrenamiento y 5,000 para validación.
* **Resolución de entrada:** 640×640 píxeles.
* **Costo computacional de entrenamiento:** Entre 10 y 12 horas por época en GPU dedicada.
* **Criterio de convergencia:**
  * **YOLOv8-Conv2D:** Entrenado durante **50 épocas** (alcanzó estabilidad antes de iniciar sobreajuste).
  * **YOLOv8-Separable:** Entrenado durante **100 épocas** (requirió mayor número de iteraciones dada su menor capacidad representacional por capa).

---

## 📈 Resultados Experimentales

### 1. Desglose de Parámetros de los Modelos

![Tabla 3 - Comparación de Parámetros](outputs/figures/tabla%203.png)

| Métrica de Arquitectura | YOLOv8-Conv2D | YOLOv8-Separable | Reducción |
| :--- | :--- | :--- | :--- |
| **Parámetros Totales** | 3,991,584 (15.23 MB) | **1,258,715 (4.80 MB)** | **-68.46%** |
| **Parámetros Entrenables** | 3,978,688 (15.18 MB) | **1,245,819 (4.75 MB)** | **-68.68%** |
| **Parámetros No Entrenables** | 12,896 (50.38 KB) | 12,896 (50.38 KB) | 0% |

---

### 2. Métricas de Calidad de Detección y Eficiencia Computacional

![Tabla de Resultados de Validación Tesis](outputs/figures/tabla%20resultados%20tesis.png)

| Plataforma de Prueba | Modelo | FPS | Tiempo Promedio por Imagen ($\overline{t} \pm \sigma$) |
| :--- | :--- | :---: | :---: |
| **PC - GPU (RTX 4060 Ti)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | **5.48 FPS**<br>5.36 FPS | $0.1824 \pm 0.0169\text{ s}$<br>$0.1866 \pm 0.1671\text{ s}$ |
| **PC - CPU (Ryzen 7)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 2.34 FPS<br>**2.41 FPS** | $0.4273 \pm 0.0207\text{ s}$<br>$0.4156 \pm 0.0136\text{ s}$ |
| **Raspberry Pi 4** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 0.07 FPS<br>**0.68 FPS** | $13.9322 \pm 1.5230\text{ s}$<br>**$1.4754 \pm 0.0142\text{ s}$** |

---

## 📊 Visualizaciones de Resultados

### Evolución de la Métrica mAP por Épocas

![Evolución mAP](outputs/figures/mAP.png)

* **Gráfico interactivo:** [Análisis Trade-off mAP vs Latencia](https://htmlpreview.github.io/?https://github.com/Rxyxs/yolov8-separable-convolutions/blob/main/outputs/interactive/latency_map_tradeoff.html)

---

## 💡 Conclusiones Principales

* **Entornos con aceleración dedicada (GPU / CPU potente):** El modelo estándar **YOLOv8-Conv2D** ofrece el mejor desempeño en mAP50, IoU y Recall. En estos entornos, las diferencias en velocidad de inferencia son marginales (5.48 vs 5.36 FPS) debido a que los Tensor Cores absorben el costo aritmético de la convolución 2D.
* **Entornos embebidos de capacidad reducida (Raspberry Pi 4):** El beneficio del modelo **YOLOv8-Separable** es drástico, multiplicando la velocidad de inferencia por casi un orden de magnitud (de 0.07 FPS a 0.68 FPS) y reduciendo la latencia de 13.93 s a 1.47 s por cuadro, con una variabilidad de tiempo ($\sigma$) sumamente estable.
* **Compromiso precisión vs. velocidad:** Existe un *trade-off* directo en el que YOLOv8-Separable sacrifica un ~3.68% de mAP50 a cambio de permitir la operatividad práctica en hardware embebido sin aceleración.

---

## 🚀 Futuras Líneas de Investigación

* **Hardware especializado:** Probar el despliegue en aceleradores de bajo consumo como Google Coral Edge TPU, NVIDIA Jetson Nano/Orin y NPU/FPGAs.
* **Recuperación de precisión:** Aplicar técnicas de ajuste fino (*fine-tuning*) progresivo y cuantización (*INT8 / FP16*) para recuperar la pérdida de mAP en modelos separables.
* **Comparativa ampliada:** Evaluar contra otras arquitecturas livianas nativas como NanoDet, MobileNet-SSD y PP-YOLO.
* **Pruebas en tiempo real:** Medir consumo energético (Watts), temperatura de trabajo y estabilidad de flujo continuo en escenarios operativos reales.

---

## 📜 Referencia Académica y Créditos

Si utilizas este trabajo o código en tu investigación, favor citar:

```bibtex
@thesis{reyespino2026yolov8,
  author      = {Pablo Vicente Reyes Pino},
  title       = {Evaluación de la eficiencia de YOLOv8 en GPU, CPU y Raspberry Pi: Convoluciones estándar vs separables},
  school      = {Universidad Mayor},
  faculty     = {Facultad de Ciencias, Ingeniería y Tecnología},
  department  = {Escuela de Ingeniería Civil en Computación e Informática},
  advisor     = {Dr. Anthony D. Cho},
  address     = {Santiago, Chile},
  year        = {2026}
}
