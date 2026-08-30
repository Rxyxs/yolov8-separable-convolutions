[ 🇺🇸 English ] | [ 🇨🇱 [Leer en Español](README.es.md) ]

# Evaluating YOLOv8 Efficiency on GPU, CPU, and Raspberry Pi: Standard vs. Separable Convolutions

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-CUDA%20%7C%20TensorRT-FF6F00?logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-XNNPACK-FF6F00?logo=tensorflow&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi%204-edge%20deployment-A22846?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> **Thesis Project:** Evaluating the trade-off between object-detection performance and computational efficiency in the YOLOv8 model.

---

## 📌 Overview

This repository contains the structure, methodology, and experimental results developed for the thesis *"Evaluación de la eficiencia de YOLOv8 en GPU, CPU y Raspberry Pi: Convoluciones estándar vs separables"* (Universidad Mayor, Chile).

The research's central goal is to analyze and evaluate the **YOLOv8** object-detection architecture deployed under computationally constrained environments. It compares the baseline version with traditional 2D convolutions (**YOLOv8-Conv2D**) against a variant adapted with depthwise-separable convolutions (**YOLOv8-Separable**) in the Head module.

---

## 🎯 Project Objectives

* **Analyze the architecture:** Study YOLOv8's core components (Backbone, Neck, and Head) and their optimizations.
* **Develop variants:** Implement both the classic YOLOv8 architecture and the modified adaptation with depthwise-separable convolution layers.
* **Experimental evaluation:** Compare precision, recall, IoU, mAP, inference speed, and resource-consumption metrics.
* **Deployment on constrained hardware:** Determine operational viability on embedded platforms (Raspberry Pi 4) versus desktop computers (GPU/CPU).

---

## 🛠️ Hardware Environments Evaluated

| Specification | Desktop PC | Raspberry Pi 4 |
| :--- | :--- | :--- |
| **Processor** | AMD Ryzen 7 5700X (8C/16T, 3.4 GHz) | Broadcom BCM2711 (ARM Cortex-A72, 1.5 GHz) |
| **GPU** | NVIDIA RTX 4060 Ti (8 GB GDDR6) | No dedicated GPU (CPU-only processing) |
| **RAM** | 32 GB DDR4 | 8 GB LPDDR4 |
| **Framework / Runtime** | TensorFlow / Keras (CUDA / TensorRT) | TensorFlow Lite / XNNPACK |

---

## 📊 Dataset Used

* **Database:** COCO2017.
* **Training:** 118,287 images.
* **Validation:** 5,000 images.
* **Categories:** 80 labeled object classes.
* **Input resolution:** **640×640** pixels.

---

## 📈 Summary Results

### 1. Parameters and Detection-Quality Metrics

| Metric / Parameter | YOLOv8-Conv2D | YOLOv8-Separable |
| :--- | :--- | :--- |
| **Total Parameters** | 3,991,584 (15.23 MB) | **1,258,715 (4.80 MB)** |
| **Training Epochs** | 50 epochs | 100 epochs |
| **mAP50** | **0.2119** | 0.1751 |
| **Average IoU** | **0.8367** | 0.8302 |
| **Precision** | **0.6836** | 0.6734 |
| **Recall** | **0.3260** | 0.2933 |

---

### 2. Inference Performance and Speed

| Test Environment | Model | FPS | Avg. Time per Image ($\overline{t} \pm \sigma$) |
| :--- | :--- | :--- | :--- |
| **PC - GPU (RTX 4060 Ti)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | **5.48 FPS**<br>5.36 FPS | $0.1824 \pm 0.0169\text{ s}$<br>$0.1866 \pm 0.1671\text{ s}$ |
| **PC - CPU (Ryzen 7)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 2.34 FPS<br>**2.41 FPS** | $0.4273 \pm 0.0207\text{ s}$<br>$0.4156 \pm 0.0136\text{ s}$ |
| **Raspberry Pi 4** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 0.07 FPS<br>**0.68 FPS** | $13.9322 \pm 1.5230\text{ s}$<br>**$1.4754 \pm 0.0142\text{ s}$** |

---

## 💡 Key Conclusions

* **Performance on accelerated infrastructure:** The standard-convolution version (**YOLOv8-Conv2D**) achieves superior detection quality (higher mAP and Recall). On desktop machines with a GPU or a powerful CPU, inference-time differences are marginal.
* **Parameter optimization:** Replacing standard convolutions with depthwise-separable ones (**YOLOv8-Separable**) reduces the total parameter count by approximately **68%**.
* **Impact on embedded devices:** In environments without dedicated acceleration, like the Raspberry Pi, **YOLOv8-Separable** increases execution speed by almost an order of magnitude (from 0.07 to 0.68 FPS) and delivers more stable frame-to-frame latency.

---

## License

MIT — see [LICENSE](LICENSE).

## 📜 Credits and Academic Reference

* **Author:** Pablo Vicente Reyes Pino
* **Advisor:** Dr. Anthony D. Cho
* **Institution:** Universidad Mayor — Escuela de Ingeniería Civil en Computación e Informática
* **Location and date:** Santiago, Chile — April 2026
