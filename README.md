# EVALUACIÓN DE LA EFICIENCIA DE YOLOV8 EN GPU, CPU Y RASPBERRY PI: CONVOLUCIONES ESTÁNDAR VS SEPARABLES

> **Autor:** Pablo Vicente Reyes Pino  
> **Profesor Guía:** Dr. Anthony D. Cho | **Profesor Revisor:** Carlos Muñoz  
> **Institución:** Universidad Mayor — Escuela de Ingeniería Civil en Computación e Informática  

---

## 1. INTRODUCCIÓN

### Motivación y Justificación
- Explicación de la necesidad de visión computacional en tiempo real y la barrera del costo computacional en hardware restringido.

### Descripción del Problema
- Brecha entre el alto consumo de recursos de Deep Learning y las limitaciones de memoria/procesamiento en sistemas embebidos.

![Figura 1: Ejemplo de detección de objetos (YOLO)](docs/images/figura1_ejemplo_yolo.png)

---

## 2. OBJETIVOS E HIPÓTESIS

### Objetivo General
- Analizar y evaluar el modelo YOLOv8 para la detección de objetos, implementándolo en un entorno práctico en hardware reducido.

### Objetivos Específicos
- Analizar la arquitectura y mejoras introducidas en YOLOv8.
- Desarrollar las versiones clásica (Conv2D) y modificada (SeparableConv2D).
- Evaluar el rendimiento en precisión, velocidad y eficiencia computacional.
- Comparar el desempeño en hardware de capacidad reducida.

### Hipótesis de Investigación
> Los modelos de Deep Learning adaptados para hardware reducido tendrán una precisión mayor que modelos de Deep Learning tradicionales en función de los recursos disponibles, comparando métricas específicas de rendimiento.

---

## 3. MARCO TEÓRICO

### Arquitectura de YOLOv8
- **Backbone:** CSPDarknet y extracción de características multiescala.
- **Neck:** FPN y PANet para fusión de características.
- **Head:** Estrategia desacoplada de clasificación y regresión de cajas.
- **Funciones de Activación:** Matemáticas de la función SiLU.

![Figura 2: Arquitectura de YOLOv8 (Backbone-Neck-Head)](docs/images/figura2_arquitectura_yolov8.png)

### Conv2D vs Convoluciones Separables (Depthwise + Pointwise)
- Detalle matemático de operaciones y reducción paramétrica.

![Figura 3: Conv2D vs SeparableConv2D](docs/images/figura3_conv2d_vs_separable.png)
![Figura 4: Ejemplo de procesos convolución estándar vs convolución separable](docs/images/figura4_procesos_convolucion.png)

---

## 4. METODOLOGÍA

### Diseño Experimental Comparativo
![Figura 5: Proceso de entrenamiento/validación](docs/images/figura5_proceso_entrenamiento.png)

### Dataset COCO2017
| Conjunto | Imágenes | Formato de Anotación |
| :--- | :--- | :--- |
| **Train** | 118,287 | Coordenadas $(c_x, c_y, w, h)$ |
| **Val** | 5,000 | Coordenadas $(c_x, c_y, w, h)$ |

### Hardware de Prueba
| Componente | PC Escritorio | Raspberry Pi 4 |
| :--- | :--- | :--- |
| **Procesador** | AMD Ryzen 7 5700X (8C/16T, 3.4GHz) | Broadcom BCM2711 (Quad-core Cortex-A72, 1.5GHz) |
| **GPU** | NVIDIA RTX 4060 Ti (8GB GDDR6) | Sin GPU dedicada (CPU NEON) |
| **RAM** | 32 GB DDR4 | 8 GB LPDDR4 |

### Métricas de Evaluación
- Definiciones y fórmulas matemáticas de **Precision**, **Recall**, **IoU**, **AP**, **mAP50** y **mAP50-95**.

---

## 5. DESARROLLO Y RESULTADOS

### Comparación de Parámetros del Modelo
| Modelo | Parámetros Totales | Parámetros Entrenables | Parámetros No Entrenables |
| :--- | :--- | :--- | :--- |
| **YOLOv8 Modificado (Separable)** | 1,258,715 (4.80 MB) | 1,245,819 (4.75 MB) | 12,896 (50.38 KB) |
| **YOLOv8 Base (Conv2D)** | 3,991,584 (15.23 MB) | 3,978,688 (15.18 MB) | 12,896 (50.38 KB) |

### Evolución del Rendimiento (mAP por época)
![Figura 6: Evolución del mAP por cada 10 épocas](docs/images/figura6_map_epocas.png)

### Resultados de Validación Cualitativos
![Figura 7: Detecciones cualitativas YOLOv8-Conv2D (50 épocas)](docs/images/figura7_deteccion_conv2d.png)
![Figura 8: Detecciones cualitativas YOLOv8-Separable (100 épocas)](docs/images/figura8_deteccion_separable.png)

### Tabla General de Resultados de Validación e Inferencia
| Métrica / Dispositivo | YOLOv8-Separable (100 Epoch) | YOLOv8-Conv2D (50 Epoch) |
| :--- | :--- | :--- |
| **mAP50** | 0.1751 | 0.2119 |
| **IoU Promedio** | 0.8302 | 0.8367 |
| **Precisión** | 0.6734 | 0.6836 |
| **Recall** | 0.2933 | 0.3260 |
| **FPS en CPU** | 2.41 | 2.34 |
| **FPS en GPU** | 5.36 | 5.48 |
| **FPS en Raspberry Pi** | **0.68** | **0.07** |
| **Tiempo Promedio Raspberry Pi (s)** | **1.4754 ± 0.0142** | **13.9322 ± 1.5230** |

---

## 6. CONCLUSIONES Y FUTURAS LÍNEAS DE INVESTIGACIÓN

### Conclusiones
- Análisis detallado del *trade-off* entre precisión y eficiencia computacional.
- Evaluación del impacto de la reducción del 68% de parámetros en hardware restringido vs. plataformas aceleradas.

### Futuras Líneas de Investigación
- Despliegue en hardware especializado (Google Edge TPU, NVIDIA Jetson, FPGAs/NPUs).
- Comparativa frente a detectores ultraligeros (NanoDet, MobileNet-SSD).
- Aplicación de técnicas de cuantización (INT8) y poda de parámetros (*pruning*).
