# Evaluación de la Eficiencia de YOLOv8 en GPU, CPU y Raspberry Pi: Convoluciones Estándar vs. Separables

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-CUDA%20%7C%20TensorRT-FF6F00?logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-XNNPACK-FF6F00?logo=tensorflow&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi%204-Edge%20Deployment-A22846?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Proyecto de Tesis:** Evaluación del compromiso (*trade-off*) entre calidad de detección de objetos y eficiencia computacional en arquitecturas YOLOv8 en entornos restringidos.

---

## Datos Académicos

* **Autor:** Pablo Vicente Reyes Pino
* **Profesor Guía:** Dr. Anthony D. Cho
* **Profesor Revisor:** Carlos Muñoz
* **Institución:** Escuela de Ingeniería Civil en Computación e Informática, Facultad de Ciencias, Ingeniería y Tecnología, Universidad Mayor (Santiago, Chile)
* **Fecha:** Abril / Julio 2026

---

## Estructura del Proyecto

```text
yolov8-separable-convolutions/
├── outputs/
│   └── figures/
│       ├── CONV2_50Epoch.png                 # Muestra de inferencia visual YOLOv8-Conv2D (50 épocas)
│       ├── SEP_100EPOCH.png                  # Muestra de inferencia visual YOLOv8-Separable (100 épocas)
│       ├── comparacion modelos.png           # Cuadro comparativo del Estado del Arte
│       ├── comparcion.png                    # Gráfico de barras de reducción de parámetros
│       ├── fps_by_device.png                 # Rendimiento en cuadros por segundo (FPS) por dispositivo
│       ├── latency_by_device.png             # Latencia de inferencia (ms/s) por dispositivo
│       ├── mAP.png                           # Curvas de convergencia mAP50 durante el entrenamiento
│       ├── map_vs_latency_tradeoff.png       # Análisis del compromiso Precisión vs Latencia
│       ├── model_size_comparison.png         # Comparativa de tamaño de modelos en disco (MB)
│       ├── tabla 3.png                       # Resumen paramétrico de las arquitecturas
│       ├── tabla resultados tesis.png        # Consolidation table con métricas de la tesis
│       ├── training_convergence_separable.png# Curva detallada de pérdida y convergencia del modelo separable
│       └── yolo_arq.pdf                      # Esquema arquitectónico detallado de YOLOv8
├── models/                                   # Pesos y archivos exportados (.pt, .onnx, .tflite)
├── scripts/                                  # Scripts de entrenamiento, exportación y evaluación de inferencia
├── README.md                                 # Documentación principal del repositorio
└── LICENSE                                   # Licencia del proyecto
