# Evaluación de la Eficiencia de YOLOv8 en GPU, CPU y Raspberry Pi: Convoluciones Estándar vs. Separables

> **Proyecto de Tesis:** Evaluación del compromiso entre desempeño en detección de objetos y eficiencia computacional del modelo YOLOv8.

---

## 📌 Descripción General

Este repositorio contiene la estructura, metodología y resultados experimentales desarrollados en la tesis *"Evaluación de la eficiencia de YOLOv8 en GPU, CPU y Raspberry Pi: Convoluciones estándar vs separables"*.

El objetivo central de la investigación es analizar y evaluar la arquitectura de detección de objetos **YOLOv8** implementada en entornos con restricciones computacionales. Se realiza una comparación entre la versión base con convoluciones 2D tradicionales (**YOLOv8-Conv2D**) y una variante adaptada mediante el uso de convoluciones separables en profundidad (**YOLOv8-Separable**) en el módulo Head.

---

## 🎯 Objetivos del Proyecto

* **Analizar la arquitectura:** Estudiar los componentes base de YOLOv8 (Backbone, Neck y Head) y sus optimizaciones.
* **Desarrollar variantes:** Implementar la arquitectura clásica de YOLOv8 y la adaptación modificada con capas de convolución separable en profundidad.
* **Evaluación experimental:** Comparar el desempeño en métricas de precisión, recall, IoU, mAP, velocidad de inferencia y consumo de recursos.
* **Despliegue en hardware reducido:** Determinar la viabilidad operativa en plataformas embebidas (Raspberry Pi 4) frente a computadores de escritorio (GPU/CPU).

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

### 1. Parámetros y Métricas de Calidad de Detección

| Métrica / Parámetro | YOLOv8-Conv2D | YOLOv8-Separable |
| :--- | :--- | :--- |
| **Parámetros Totales** | 3,991,584 (15.23 MB) | **1,258,715 (4.80 MB)** |
| **Épocas de Entrenamiento** | 50 épocas | 100 épocas |
| **mAP50** | **0.2119** | 0.1751 |
| **IoU Promedio** | **0.8367** | 0.8302 |
| **Precisión** | **0.6836** | 0.6734 |
| **Sensibilidad (Recall)** | **0.3260** | 0.2933 |

---

### 2. Rendimiento y Velocidad de Inferencia

| Entorno de Prueba | Modelo | FPS | Tiempo Promedio por Imagen ($\overline{t} \pm \sigma$) |
| :--- | :--- | :--- | :--- |
| **PC - GPU (RTX 4060 Ti)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | **5.48 FPS**<br>5.36 FPS | $0.1824 \pm 0.0169\text{ s}$<br>$0.1866 \pm 0.1671\text{ s}$ |
| **PC - CPU (Ryzen 7)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 2.34 FPS<br>**2.41 FPS** | $0.4273 \pm 0.0207\text{ s}$<br>$0.4156 \pm 0.0136\text{ s}$ |
| **Raspberry Pi 4** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 0.07 FPS<br>**0.68 FPS** | $13.9322 \pm 1.5230\text{ s}$<br>**$1.4754 \pm 0.0142\text{ s}$** |

---

## 💡 Conclusiones Clave

* **Desempeño en Infraestructura con Aceleración:** La versión con convoluciones estándar (**YOLOv8-Conv2D**) logra una calidad de detección superior (mayor mAP y Recall). En equipos de escritorio con GPU o CPU potente, las diferencias en tiempo de inferencia son marginales.
* **Optimización de Parámetros:** El reemplazo por convoluciones separables en profundidad (**YOLOv8-Separable**) reduce el número total de parámetros en aproximadamente un **68%**.
* **Impacto en Dispositivos Embebidos:** En entornos sin aceleración dedicada como la Raspberry Pi, **YOLOv8-Separable** incrementa la velocidad de ejecución en casi un orden de magnitud (de 0.07 a 0.68 FPS) y garantiza mayor estabilidad en la latencia cuadro a cuadro.

---

## 📜 Créditos y Referencia Académica

* **Autor:** Pablo Vicente Reyes Pino
* **Tutor:** Dr. Anthony D. Cho
* **Institución:** Universidad Mayor — Escuela de Ingeniería Civil en Computación e Informática
* **Ubicación y Fecha:** Santiago, Chile — Abril 2026
