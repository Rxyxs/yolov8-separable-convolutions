# Evaluación de la Eficiencia de YOLOv8 en GPU, CPU y Raspberry Pi: Convoluciones Estándar vs. Separables

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-CUDA%20%7C%20TensorRT-FF6F00?logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-XNNPACK-FF6F00?logo=tensorflow&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi%204-Edge%20Deployment-A22846?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Proyecto de Tesis:** Evaluación del compromiso (*trade-off*) entre calidad de detección de objetos y eficiencia computacional en arquitecturas YOLOv8 en entornos restringidos[cite: 1, 2].

---

## 📌 Datos Académicos

* **Autor:** Pablo Vicente Reyes Pino[cite: 1, 2].
* **Profesor Guía:** Dr. Anthony D. Cho[cite: 1, 2].
* **Profesor Revisor:** Carlos Muñoz[cite: 1].
* **Institución:** Escuela de Ingeniería Civil en Computación e Informática, Facultad de Ciencias, Ingeniería y Tecnología, Universidad Mayor (Santiago, Chile)[cite: 1, 2].
* **Fecha:** Abril / Julio 2026[cite: 1, 2].

---

## 📌 Resumen y Motivación

La detección de objetos en tiempo real es fundamental en sistemas de visión artificial modernos, como conducción autónoma, videovigilancia y robótica industrial[cite: 1, 2]. No obstante, los modelos de aprendizaje profundo tradicionales requieren una elevada capacidad de procesamiento que dificulta su despliegue en dispositivos de borde o hardware reducido[cite: 1, 2].

Esta investigación evalúa el comportamiento del modelo **YOLOv8** modificando su módulo de detección (*Head*) mediante la sustitución de convoluciones 2D estándar (**YOLOv8-Conv2D**) por convoluciones separables en profundidad (**YOLOv8-Separable**)[cite: 1, 2]. Ambos modelos fueron probados bajo un estricto protocolo experimental idéntico a lo largo de tres perfiles de hardware distintos: GPU dedicada, CPU de escritorio y una plataforma embebida Raspberry Pi 4[cite: 1, 2].

---

## 🔬 Hipótesis de Investigación

> *"Los modelos de Deep Learning adaptados para hardware reducido tendrán una precisión mayor que los modelos de Deep Learning tradicionales en función de los recursos disponibles, comparando métricas específicas de rendimiento."*[cite: 1, 2]

---

## 🧭 Marco Teórico y Modificación Arquitectónica

El modelo YOLOv8 organiza su flujo en tres módulos principales: **Backbone** (extracción de características con CSPDarknet y activación SiLU), **Neck** (fusión multiescala vía FPN y PANet con bloques C2f y SPPF) y **Head** (generación de predicciones de cajas y clases)[cite: 2].

### Convolución Conv2D (Estándar)
Aplica un filtro tridimensional completo sobre los canales de entrada[cite: 1, 2]. El número de parámetros requeridos para un kernel de tamaño $k$ con $C_{\text{in}}$ canales de entrada y $C_{\text{out}}$ filtros de salida es:
$$P_{\text{conv\_std}} = k \times k \times C_{\text{in}} \times C_{\text{out}}$$[cite: 2]

### Convolución Separable en Profundidad
Factoriza la operación convolucional en dos etapas independientes[cite: 1, 2]:
1. **Convolución Depthwise:** Aplica un filtro espacial $k \times k$ a cada canal de entrada de forma individual[cite: 2].
   $$P_{\text{depthwise}} = k \times k \times C_{\text{in}}$$[cite: 2]
2. **Convolución Pointwise:** Realiza una proyección lineal $1 \times 1$ para mezclar la información entre canales[cite: 2].
   $$P_{\text{pointwise}} = C_{\text{in}} \times C_{\text{out}}$$[cite: 2]

Esta sustitución estratégica en el módulo *Head* reduce la complejidad computacional y el número de parámetros del modelo en aproximadamente un **68%**[cite: 1, 2].

---

## 🛠️ Entornos de Hardware y Justificación de Plataformas

| Especificación | PC de Escritorio (GPU) | PC de Escritorio (CPU) | Raspberry Pi 4 | ESP32-CAM |
| :--- | :--- | :--- | :--- | :--- |
| **Procesador** | AMD Ryzen 7 5700X (8C/16T, 3.4 GHz)[cite: 1, 2] | AMD Ryzen 7 5700X[cite: 1, 2] | Broadcom BCM2711 (Quad-core ARM Cortex-A72 @ 1.5 GHz)[cite: 1, 2] | ESP32-D0WDQ6 Dual-core @ 240 MHz[cite: 2] |
| **Acelerador Gráfico** | NVIDIA RTX 4060 Ti (8 GB GDDR6)[cite: 1, 2] | Sin GPU dedicada[cite: 2] | Sin GPU dedicada (ARM NEON SIMD)[cite: 1, 2] | Sin acelerador[cite: 2] |
| **Memoria RAM** | 32 GB DDR4 3200 MHz[cite: 1, 2] | 32 GB DDR4[cite: 1, 2] | 8 GB LPDDR4 2133 MHz[cite: 1, 2] | 520 KB SRAM + 4 MB PSRAM[cite: 2] |
| **Consumo Energético** | ~160W (GPU) + 65W (CPU)[cite: 2] | ~65W[cite: 2] | **~7W**[cite: 1, 2] | Ultra bajo[cite: 2] |
| **Runtime / Framework** | TensorFlow / Keras (CUDA / TensorRT)[cite: 1, 2] | TensorFlow CPU[cite: 1, 2] | TensorFlow Lite / XNNPACK[cite: 1, 2] | Incompatible[cite: 2] |
| **Estado de Despliegue** | Evaluado[cite: 1, 2] | Evaluado[cite: 1, 2] | **Evaluado**[cite: 1, 2] | **Descartado** (memoria insuficiente para cargar el grafo de YOLOv8)[cite: 2] |

---

## 📊 Configuración del Dataset y Entrenamiento

* **Base de datos:** COCO2017 (80 categorías de objetos etiquetados)[cite: 1, 2].
* **División de datos:** 118,287 imágenes para entrenamiento y 5,000 para validación[cite: 1, 2].
* **Resolución de entrada:** 640×640 píxeles[cite: 1, 2].
* **Costo computacional de entrenamiento:** Entre 10 y 12 horas por época en GPU dedicada[cite: 1, 2].
* **Criterio de convergencia:**
  * **YOLOv8-Conv2D:** Entrenado durante **50 épocas** (alcanzó estabilidad antes de iniciar sobreajuste)[cite: 1, 2].
  * **YOLOv8-Separable:** Entrenado durante **100 épocas** (requirió mayor número de iteraciones dada su menor capacidad representacional por capa)[cite: 1, 2].

---

## 📈 Resultados Experimentales

### 1. Desglose de Parámetros de los Modelos

| Métrica de Arquitectura | YOLOv8-Conv2D | YOLOv8-Separable | Reducción |
| :--- | :--- | :--- | :--- |
| **Parámetros Totales** | 3,991,584 (15.23 MB)[cite: 2] | **1,258,715 (4.80 MB)**[cite: 2] | **-68.46%**[cite: 1, 2] |
| **Parámetros Entrenables** | 3,978,688 (15.18 MB)[cite: 2] | **1,245,819 (4.75 MB)**[cite: 2] | **-68.68%**[cite: 2] |
| **Parámetros No Entrenables** | 12,896 (50.38 KB)[cite: 2] | 12,896 (50.38 KB)[cite: 2] | 0%[cite: 2] |

### 2. Métricas de Calidad de Detección (Validación COCO2017)

| Modelo | Épocas | mAP50 | IoU Promedio | Precisión | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8-Conv2D** | 50 | **0.2119**[cite: 1, 2] | **0.8367**[cite: 1, 2] | **0.6836**[cite: 1, 2] | **0.3260**[cite: 1, 2] |
| **YOLOv8-Separable** | 100 | 0.1751[cite: 1, 2] | 0.8302[cite: 1, 2] | 0.6734[cite: 1, 2] | 0.2933[cite: 1, 2] |

### 3. Benchmarks de Inferencia por Entorno de Hardware

| Plataforma de Prueba | Modelo | FPS | Tiempo Promedio por Imagen ($\overline{t} \pm \sigma$) |
| :--- | :--- | :---: | :---: |
| **PC - GPU (RTX 4060 Ti)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | **5.48 FPS**[cite: 1, 2]<br>5.36 FPS[cite: 1, 2] | $0.1824 \pm 0.0169\text{ s}$[cite: 1, 2]<br>$0.1866 \pm 0.1671\text{ s}$[cite: 1, 2] |
| **PC - CPU (Ryzen 7)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 2.34 FPS[cite: 1, 2]<br>**2.41 FPS**[cite: 1, 2] | $0.4273 \pm 0.0207\text{ s}$[cite: 1, 2]<br>$0.4156 \pm 0.0136\text{ s}$[cite: 1, 2] |
| **Raspberry Pi 4** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 0.07 FPS[cite: 1, 2]<br>**0.68 FPS**[cite: 1, 2] | $13.9322 \pm 1.5230\text{ s}$[cite: 1, 2]<br>**$1.4754 \pm 0.0142\text{ s}$**[cite: 1, 2] |

---

## 📊 Visualizaciones de Resultados

| Curva de Convergencia mAP | Detección Cualitativa (Conv2D) |
| :---: | :---: |
| ![Evolución mAP](outputs/figures/training_convergence_separable.png)[cite: 1, 2] | ![Detección Conv2D](outputs/figures/model_size_comparison.png)[cite: 1, 2] |

* **Gráfico interactivo:** [Análisis Trade-off mAP vs Latencia](https://htmlpreview.github.io/?https://github.com/Rxyxs/yolov8-separable-convolutions/blob/main/outputs/interactive/latency_map_tradeoff.html)

---

## 💡 Conclusiones Principales

* **Entornos con aceleración dedicada (GPU / CPU potente):** El modelo estándar **YOLOv8-Conv2D** ofrece el mejor desempeño en mAP50, IoU y Recall[cite: 1, 2]. En estos entornos, las diferencias en velocidad de inferencia son marginales (5.48 vs 5.36 FPS) debido a que los Tensor Cores absorben el costo aritmético de la convolución 2D[cite: 1, 2].
* **Entornos embebidos de capacidad reducida (Raspberry Pi 4):** El beneficio del modelo **YOLOv8-Separable** es drástico, multiplicando la velocidad de inferencia por casi un orden de magnitud (de 0.07 FPS a 0.68 FPS) y reduciendo la latencia de 13.93 s a 1.47 s por cuadro, con una variabilidad de tiempo ($\sigma$) sumamente estable[cite: 1, 2].
* **Compromiso precisión vs. velocidad:** Existe un *trade-off* directo en el que YOLOv8-Separable sacrifica un ~3.68% de mAP50 a cambio de permitir la operatividad práctica en hardware embebido sin aceleración[cite: 1, 2].

---

## 🚀 Futuras Líneas de Investigación

* **Hardware especializado:** Probar el despliegue en aceleradores de bajo consumo como Google Coral Edge TPU, NVIDIA Jetson Nano/Orin y NPU/FPGAs[cite: 1].
* **Recuperación de precisión:** Aplicar técnicas de ajuste fino (*fine-tuning*) progresivo y cuantización (*INT8 / FP16*) para recuperar la pérdida de mAP en modelos separables[cite: 1].
* **Comparativa ampliada:** Evaluar contra otras arquitecturas livianas nativas como NanoDet, MobileNet-SSD y PP-YOLO[cite: 1].
* **Pruebas en tiempo real:** Medir consumo energético (Watts), temperatura de trabajo y estabilidad de flujo continuo en escenarios operativos reales[cite: 1].

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
```[cite: 1, 2]

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para obtener más información.
