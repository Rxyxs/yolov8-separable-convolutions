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

## Resumen y Motivación

La detección de objetos en tiempo real es fundamental en sistemas de visión artificial modernos, como conducción autónoma, videovigilancia y robótica industrial. No obstante, los modelos de aprendizaje profundo tradicionales requieren una elevada capacidad de procesamiento que dificulta su despliegue en dispositivos de borde (*edge computing*) o hardware de capacidades reducidas.

Esta investigación evalúa el comportamiento del modelo **YOLOv8** modificando su módulo de detección (*Head*) mediante la sustitución de convoluciones 2D estándar (**YOLOv8-Conv2D**) por convoluciones separables en profundidad (**YOLOv8-Separable**). Ambos modelos fueron probados bajo un estricto protocolo experimental idéntico a lo largo de tres perfiles de hardware distintos: GPU dedicada, CPU de escritorio y una plataforma embebida Raspberry Pi 4.

---

## Estado del Arte: Comparación de Modelos de Detección

Para justificar la elección de YOLO frente a otras alternativas, se analizó la evolución técnica de los modelos de visión:

![Tabla Comparativa de Modelos](outputs/figures/comparacion%20modelos.png)

> Modelos clásicos como SIFT y HOG dependen de la extracción manual de características, lo que limita su capacidad ante alta variabilidad. Si bien las arquitecturas R-CNN y sus derivadas introdujeron una alta precisión mediante propuestas de regiones, su diseño en múltiples etapas genera cuellos de botella computacionales. YOLO transformó este paradigma al realizar la detección en una sola pasada (one-stage detector), permitiendo procesamiento en tiempo real. Esta tesis busca mantener esta ventaja táctica de YOLO pero mitigando su demanda de recursos para hacerlo viable en la capa física de IoT.

---

## Hipótesis de Investigación

> *"Los modelos de Deep Learning adaptados para hardware reducido tendrán una precisión mayor que los modelos de Deep Learning tradicionales en función de los recursos disponibles, comparando métricas específicas de rendimiento."*

---

## Marco Teórico y Modificación Arquitectónica

### Análisis de la Arquitectura Base (YOLOv8)

De acuerdo al diagrama arquitectónico analizado[cite: 3], el modelo YOLOv8 estructura su flujo de datos en tres macrosecciones interconectadas:

1. **Backbone:** Encargado de procesar la entrada inicial de dimensiones $640\times640\times3$. Utiliza una secuencia de capas ConvModule y cuellos de botella (DarknetBottleneck) integrados en bloques C2f, culminando en un módulo SPPF para agrupar características espaciales[cite: 3].
2. **Neck:** Integra las características multiescala extraídas en las etapas P3, P4 y P5[cite: 3]. Mediante operaciones Upsample, Concat y bloques C2f adicionales, fusiona la semántica profunda con los detalles espaciales de las primeras capas[cite: 3].
3. **Head:** Se bifurca en dos ramas independientes para cada escala ($80\times80$, $40\times40$ y $20\times20$)[cite: 3]. Una rama computa la pérdida de clasificación (Cls Loss BCE) y la otra calcula la regresión de cajas delimitadoras (Bbox Loss CIoU + DFL)[cite: 3]. 

La intervención arquitectónica propuesta en esta investigación se aplica **únicamente en las capas Conv2d terminales del módulo Head**[cite: 3], sustituyéndolas por convoluciones separables para reducir la carga de parámetros sin alterar la extracción de características del Backbone y Neck[cite: 3].

### Modificación Matemática

**Convolución Conv2D (Estándar):**
Aplica un filtro tridimensional completo. Para un kernel de tamaño $k$ con $C_{in}$ canales de entrada y $C_{out}$ filtros de salida, el costo paramétrico es:
$$P_{estandar}=k\times k\times C_{in}\times C_{out}$$

**Convolución Separable en Profundidad:**
Factoriza la operación en dos etapas:
1. **Depthwise:** Filtro espacial $k \times k$ a cada canal individual.
   $$P_{depthwise}=k\times k\times C_{in}$$
2. **Pointwise:** Proyección lineal $1 \times 1$ para mezclar canales.
   $$P_{pointwise}=C_{in}\times C_{out}$$

---

## Entornos de Hardware y Viabilidad

| Especificación | PC de Escritorio (GPU) | PC de Escritorio (CPU) | Raspberry Pi 4 |
| :--- | :--- | :--- | :--- |
| **Procesador** | AMD Ryzen 7 5700X | AMD Ryzen 7 5700X | Broadcom BCM2711 (ARM) |
| **Acelerador Gráfico** | NVIDIA RTX 4060 Ti | Sin GPU dedicada | Sin GPU dedicada |
| **Consumo Energético** | ~160W (GPU) + 65W (CPU) | ~65W | **~7W** |
| **Runtime / Framework** | TensorFlow / Keras (CUDA) | TensorFlow CPU | TensorFlow Lite (XNNPACK) |

*(Nota: El hardware ESP32-CAM fue descartado durante las pruebas iniciales debido a desbordamientos de memoria Out-Of-Memory al intentar asignar el grafo de tensores).*

---

## Resultados Experimentales

### 1. Desglose de Parámetros de los Modelos

![Comparación de Parámetros](outputs/figures/comparcion.png)

> Al comparar la densidad de la arquitectura, se aprecia una optimización drástica. Al implementar las convoluciones separables, logramos reducir los parámetros totales en un **68.4%**, cayendo de 3.99 millones a solo 1.25 millones. Esta compresión paramétrica es la que habilita mecánicamente el despliegue en memoria RAM limitada.

---

### 2. Eficiencia Computacional en Dispositivos

| Plataforma de Prueba | Modelo | FPS | Tiempo Promedio por Imagen ($\overline{t} \pm \sigma$) |
| :--- | :--- | :---: | :---: |
| **PC - GPU (RTX 4060 Ti)** | YOLOv8-Conv2D<br>YOLOv8-Separable | 5.48 FPS<br>5.36 FPS | $0.1824 \pm 0.0169\text{ s}$<br>$0.1866 \pm 0.1671\text{ s}$ |
| **PC - CPU (Ryzen 7)** | YOLOv8-Conv2D<br>YOLOv8-Separable | 2.34 FPS<br>2.41 FPS | $0.4273 \pm 0.0207\text{ s}$<br>$0.4156 \pm 0.0136\text{ s}$ |
| **Raspberry Pi 4** | YOLOv8-Conv2D<br>**YOLOv8-Separable** | 0.07 FPS<br>**0.68 FPS** | $13.9322 \pm 1.5230\text{ s}$<br>**$1.4754 \pm 0.0142\text{ s}$** |

> El impacto crítico de esta optimización se evidencia en la Raspberry Pi. El modelo estándar resulta inoperable (14 segundos por imagen). El modelo Separable desplomó este tiempo a 1.48 segundos, una aceleración de casi 10 veces, estabilizando por completo el procesamiento en el entorno embebido.

---

### 3. Evaluación Cualitativa de Inferencia Visual

Para contrastar empíricamente el desempeño frente a la pérdida teórica de precisión (mAP), se evaluaron las inferencias visuales bajo distintas épocas de convergencia:

**Inferencia Modelo Conv2D (50 Épocas)**
![Inferencia Conv2D](outputs/figures/CONV2_50Epoch.png)

**Inferencia Modelo Separable (100 Épocas)**
![Inferencia Separable](outputs/figures/SEP_100EPOCH.png)

> **Análisis Visual:** Al requerir 100 épocas para compensar su menor expresividad matemática, el modelo Separable logra una detección funcional comparable al modelo clásico de 50 épocas. Si bien el modelo Conv2D exhibe márgenes de confianza (confidence scores) ligeramente superiores en detecciones complejas (ej. contornos de animales u objetos ocluidos), el modelo Separable mantiene la capacidad robusta de localizar y clasificar correctamente los objetos principales (personas, vehículos, señales), demostrando que el pequeño sacrificio en mAP es imperceptible para tareas operativas generales.

---

## Conclusiones Principales

* **Entornos con aceleración dedicada:** El modelo YOLOv8-Conv2D es indiscutible. La reducción de parámetros del modelo Separable no ofrece ventajas de velocidad (5.48 vs 5.36 FPS) debido a que la GPU absorbe eficientemente la carga matricial estándar.
* **Entornos embebidos restringidos:** YOLOv8-Separable es determinante. Transforma un modelo inoperable en un sistema estable, multiplicando la tasa de procesamiento por 10 en la Raspberry Pi 4.
* **Validación de la Hipótesis:** Se demuestra el *trade-off* fundamental. Asumir una caída marginal en la precisión es una estrategia arquitectónica obligatoria para destrabar cuellos de botella computacionales en dispositivos del borde.

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para obtener más información.
