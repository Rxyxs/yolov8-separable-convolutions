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

## 🧭 Problem Development

Object detection models like YOLOv8 are almost always designed and benchmarked on desktop-class GPUs, where standard 2D convolutions are cheap because dedicated tensor cores absorb their cost. In practice, however, a large share of real deployments — traffic cameras, agricultural sensors, industrial inspection lines, mobile robotics — do not have a GPU available at inference time. They run on a CPU, or on an embedded board such as a Raspberry Pi, where every multiply-accumulate operation is paid for directly in wall-clock latency and power draw.

A standard 2D convolution with kernel size `k`, `Cin` input channels and `Cout` output channels costs `k² · Cin · Cout` multiply-accumulates per output pixel. A **depthwise-separable convolution** factorizes that operation into a depthwise pass (one filter per input channel, cost `k² · Cin`) followed by a pointwise 1×1 convolution that mixes channels (cost `Cin · Cout`). The combined cost, `k² · Cin + Cin · Cout`, grows additively instead of multiplicatively with the kernel size — for a typical 3×3 kernel this removes roughly an order of magnitude of arithmetic, at the cost of a smaller function space and therefore (in general) lower representational capacity per layer.

This is exactly the trade-off this thesis sets out to measure, not assume: **does replacing standard convolutions with depthwise-separable convolutions in the YOLOv8 Head actually translate into a usable speed gain on CPU-bound and edge hardware, and how much detection quality (mAP, precision, recall) is given up in exchange?** The question matters because the theoretical FLOP reduction does not automatically produce a proportional latency reduction — memory-bandwidth limits, framework overhead, and hardware-specific kernel implementations (e.g. XNNPACK on ARM, TensorRT on GPU) all interact with the operator count in ways that only an empirical benchmark can reveal. Both architectures were trained from scratch on COCO2017 and evaluated end-to-end (same weights, same preprocessing, same evaluation protocol) across three genuinely different hardware profiles — a desktop GPU, a desktop CPU, and a Raspberry Pi 4 — so that the comparison isolates the effect of the convolution type rather than any confound from the training or evaluation setup.

---

## 🎯 Project Objectives

* **Analyze the architecture:** Study YOLOv8's core components (Backbone, Neck, and Head) and their optimizations.
* **Develop variants:** Implement both the classic YOLOv8 architecture and the modified adaptation with depthwise-separable convolution layers.
* **Experimental evaluation:** Compare precision, recall, IoU, mAP, inference speed, and resource-consumption metrics.
* **Deployment on constrained hardware:** Determine operational viability on embedded platforms (Raspberry Pi 4) versus desktop computers (GPU/CPU).

---

## 🛠️ Techniques Used

| Technique | Role in this project |
| :--- | :--- |
| **YOLOv8 architecture (Backbone / Neck / Head)** | Base object-detection architecture (`keras_cv` `yolo_v8_xs_backbone` preset), used as the common baseline for both variants. |
| **Standard 2D convolutions (YOLOv8-Conv2D)** | Control architecture — the Head module uses ordinary `Conv2D` layers, `k² · Cin · Cout` MACs/pixel. |
| **Depthwise-separable convolutions (YOLOv8-Separable)** | Treatment architecture — the Head module's convolutions are factorized into depthwise + pointwise layers (a patched `keras_cv` build), reducing parameter count and arithmetic cost. |
| **TensorFlow / Keras + CUDA / TensorRT** | Training and GPU inference runtime on the desktop PC (mirrored-strategy training, `.h5` checkpoints). |
| **TensorFlow Lite + XNNPACK** | Export/runtime path used to run inference on the Raspberry Pi 4 (CPU-only, no dedicated accelerator). |
| **COCO2017 evaluation protocol** | Training on the 118,287-image train split, evaluation on the held-out 5,000-image val split, at 640×640 input resolution, 80 object classes. |
| **Evaluation metrics** | mAP50, average IoU, precision, recall, model size (MB) / parameter count, FPS, and per-image latency (mean ± std) — the standard metric set for comparing detection quality against deployment cost. |

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

> All numbers in this section are the **original measurements produced during the thesis work** (training runs and hardware benchmarks executed on the PC and the Raspberry Pi 4 described above). They come directly from the thesis's summary tables and from the saved cell outputs of `20250131 - YOLOv8_Separable.ipynb` (e.g. the `model.summary()` output reporting `Total params: 1,258,715 (4.80 MB)`, and the 10-epoch training log). No number below was recomputed or re-benchmarked on the machine used to write this documentation — see [Reproducibility note](#-reproducibility-note) for why, and what was actually verified here.

### 1. Parameters and Detection-Quality Metrics

| Metric / Parameter | YOLOv8-Conv2D | YOLOv8-Separable |
| :--- | :--- | :--- |
| **Total Parameters** | 3,991,584 (15.23 MB) | **1,258,715 (4.80 MB)** |
| **Training Epochs** | 50 epochs | 100 epochs |
| **mAP50** | **0.2119** | 0.1751 |
| **Average IoU** | **0.8367** | 0.8302 |
| **Precision** | **0.6836** | 0.6734 |
| **Recall** | **0.3260** | 0.2933 |

### 2. Inference Performance and Speed

| Test Environment | Model | FPS | Avg. Time per Image ($\overline{t} \pm \sigma$) |
| :--- | :--- | :--- | :--- |
| **PC - GPU (RTX 4060 Ti)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | **5.48 FPS**<br>5.36 FPS | $0.1824 \pm 0.0169\text{ s}$<br>$0.1866 \pm 0.1671\text{ s}$ |
| **PC - CPU (Ryzen 7)** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 2.34 FPS<br>**2.41 FPS** | $0.4273 \pm 0.0207\text{ s}$<br>$0.4156 \pm 0.0136\text{ s}$ |
| **Raspberry Pi 4** | **YOLOv8-Conv2D**<br>**YOLOv8-Separable** | 0.07 FPS<br>**0.68 FPS** | $13.9322 \pm 1.5230\text{ s}$<br>**$1.4754 \pm 0.0142\text{ s}$** |

### 3. Real training convergence (YOLOv8-Separable, 10-epoch log)

The plot below is not a summary statistic — it is the actual epoch-by-epoch `box_loss` / `class_loss` / `val_loss` sequence saved in the notebook's own output cell for the YOLOv8-Separable training run (`Epoch 1/10 … Epoch 10/10`), included to make the reported metrics auditable against the raw training log rather than only the final numbers.

![Training convergence — YOLOv8-Separable](outputs/figures/training_convergence_separable.png)

---

## 📊 Visuals

All charts below are built directly from the tables above — no numbers were altered or estimated for the plots.

| | |
| :---: | :---: |
| ![FPS by device](outputs/figures/fps_by_device.png) | ![Latency by device](outputs/figures/latency_by_device.png) |
| ![Model size comparison](outputs/figures/model_size_comparison.png) | ![mAP vs latency trade-off](outputs/figures/map_vs_latency_tradeoff.png) |

**Interactive chart:** [Accuracy vs. Latency Trade-off Across Devices](https://htmlpreview.github.io/?https://github.com/Rxyxs/yolov8-separable-convolutions/blob/main/outputs/interactive/latency_map_tradeoff.html) — hover any point to see FPS, latency, mAP50, and parameter count for that device/architecture combination (self-contained HTML, built with Plotly from the same real benchmark table).

---

## 💡 Key Conclusions

* **Performance on accelerated infrastructure:** The standard-convolution version (**YOLOv8-Conv2D**) achieves superior detection quality (higher mAP and Recall). On desktop machines with a GPU or a powerful CPU, inference-time differences are marginal.
* **Parameter optimization:** Replacing standard convolutions with depthwise-separable ones (**YOLOv8-Separable**) reduces the total parameter count by approximately **68%**.
* **Impact on embedded devices:** In environments without dedicated acceleration, like the Raspberry Pi, **YOLOv8-Separable** increases execution speed by almost an order of magnitude (from 0.07 to 0.68 FPS) and delivers more stable frame-to-frame latency.

---

## 🔁 Reproducibility Note

The GPU/CPU/Raspberry Pi benchmarks above were produced on the specific hardware described in this thesis (Ryzen 7 5700X + RTX 4060 Ti desktop, and a physical Raspberry Pi 4), using a pinned `tensorflow==2.13.1` environment and a patched `keras_cv` build with the separable-convolution Head. Reproducing them exactly requires that same hardware/software combination plus the full COCO2017 dataset (~19 GB) and the trained model checkpoints, none of which are practical to re-run from an arbitrary documentation-editing machine — and re-running them on different hardware would not "verify" the thesis numbers, it would simply produce a different, non-comparable benchmark. For that reason, this documentation update does not attempt to reproduce or replace the GPU/CPU/Raspberry Pi figures: every number in the Results section above is the thesis's own original measurement, and the only new artifacts added here are the plots and the interactive chart, generated directly from those existing numbers (`outputs/make_figures.py`, `outputs/make_interactive.py`).

---

## License

MIT — see [LICENSE](LICENSE).

## 📜 Credits and Academic Reference

* **Author:** Pablo Vicente Reyes Pino
* **Advisor:** Dr. Anthony D. Cho
* **Institution:** Universidad Mayor — Escuela de Ingeniería Civil en Computación e Informática
* **Location and date:** Santiago, Chile — April 2026
