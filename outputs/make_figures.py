"""
Figures built from the REAL results measured during the thesis work
("Evaluacion de la eficiencia de YOLOv8 en GPU, CPU y Raspberry Pi",
Universidad Mayor, 2026). All numeric values below are copied verbatim
from README.md / README.es.md (summary result tables) and from the
saved cell outputs of `20250131 - YOLOv8_Separable.ipynb` (model.summary()
and the 10-epoch training log). Nothing here is invented.

Run: python outputs/make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Brand-neutral, colorblind-safe palette
COLOR_CONV2D = "#4C72B0"      # blue  - YOLOv8-Conv2D (standard convolutions)
COLOR_SEP = "#DD8452"         # orange - YOLOv8-Separable (depthwise-separable)
GRID = "#D9D9D9"
TEXT = "#333333"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Real data, taken directly from the thesis result tables
# ---------------------------------------------------------------------------
devices = ["GPU\n(RTX 4060 Ti)", "CPU\n(Ryzen 7 5700X)", "Raspberry Pi 4"]

fps_conv2d = [5.48, 2.34, 0.07]
fps_sep = [5.36, 2.41, 0.68]

time_conv2d = [0.1824, 0.4273, 13.9322]     # seconds/image, mean
time_conv2d_std = [0.0169, 0.0207, 1.5230]
time_sep = [0.1866, 0.4156, 1.4754]
time_sep_std = [0.1671, 0.0136, 0.0142]

params_conv2d = 3_991_584   # 15.23 MB
params_sep = 1_258_715      # 4.80 MB
size_conv2d_mb = 15.23
size_sep_mb = 4.80

mAP50_conv2d, mAP50_sep = 0.2119, 0.1751
iou_conv2d, iou_sep = 0.8367, 0.8302
precision_conv2d, precision_sep = 0.6836, 0.6734
recall_conv2d, recall_sep = 0.3260, 0.2933

# Real training log (10 epochs) copied from the saved output of cell 21 in
# "20250131 - YOLOv8_Separable.ipynb" (YOLOv8-Separable training run)
epochs = list(range(1, 11))
train_loss = [21.6492, 2.0679, 1.9203, 1.8344, 1.7813, 1.7402, 1.7093, 1.6812, 1.6580, 1.6404]
val_loss = [2.1287, 1.9751, 1.8522, 1.8059, 1.7838, 1.7225, 1.6896, 1.7003, 1.6362, 1.6301]

# ---------------------------------------------------------------------------
# Figure 1: FPS comparison across devices (log scale, since RPi differs by
# almost 2 orders of magnitude from GPU/CPU)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(devices))
w = 0.35
b1 = ax.bar(x - w / 2, fps_conv2d, w, label="YOLOv8-Conv2D (standard)", color=COLOR_CONV2D)
b2 = ax.bar(x + w / 2, fps_sep, w, label="YOLOv8-Separable (depthwise)", color=COLOR_SEP)
ax.set_yscale("log")
ax.set_ylabel("Inference speed (FPS, log scale)")
ax.set_title("Inference throughput by device and convolution type\n(real thesis benchmark, COCO2017 val, 640×640)")
ax.set_xticks(x)
ax.set_xticklabels(devices)
ax.legend(frameon=False)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fps_by_device.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Avg. inference time per image (s) with std-dev error bars, log
# scale y-axis
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w / 2, time_conv2d, w, yerr=time_conv2d_std, capsize=4,
            label="YOLOv8-Conv2D (standard)", color=COLOR_CONV2D)
b2 = ax.bar(x + w / 2, time_sep, w, yerr=time_sep_std, capsize=4,
            label="YOLOv8-Separable (depthwise)", color=COLOR_SEP)
ax.set_yscale("log")
ax.set_ylabel("Avg. inference time per image, seconds (log scale)")
ax.set_title("Per-image latency by device and convolution type\n(mean ± std, real thesis benchmark)")
ax.set_xticks(x)
ax.set_xticklabels(devices)
ax.legend(frameon=False)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "latency_by_device.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: Model size / parameter count comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5))
labels = ["YOLOv8-Conv2D\n(standard)", "YOLOv8-Separable\n(depthwise)"]
sizes = [size_conv2d_mb, size_sep_mb]
params = [params_conv2d, params_sep]
bars = ax.bar(labels, sizes, color=[COLOR_CONV2D, COLOR_SEP], width=0.55, zorder=3)
ax.set_ylabel("Model size (MB)")
ax.set_title("Model size and parameter count\n(real thesis measurements)")
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for rect, p in zip(bars, params):
    h = rect.get_height()
    ax.annotate(f"{h:.2f} MB\n({p:,} params)", (rect.get_x() + rect.get_width() / 2, h),
                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
reduction = (1 - size_sep_mb / size_conv2d_mb) * 100
ax.text(0.5, 0.78, f"-{reduction:.0f}% size / params",
        transform=ax.transAxes, ha="center", fontsize=11, color="#555555")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "model_size_comparison.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: mAP50 vs. latency trade-off across devices (the core thesis
# trade-off)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
device_markers = ["o", "s", "^"]
label_offsets = [(-8, 14), (10, 14), (-45, -18)]
for i, dev in enumerate(["GPU (RTX 4060 Ti)", "CPU (Ryzen 7)", "Raspberry Pi 4"]):
    ax.scatter(time_conv2d[i], mAP50_conv2d, s=140, color=COLOR_CONV2D,
               marker=device_markers[i], edgecolor="white", linewidth=1, zorder=3)
    ax.scatter(time_sep[i], mAP50_sep, s=140, color=COLOR_SEP,
               marker=device_markers[i], edgecolor="white", linewidth=1, zorder=3)
    ax.annotate(dev, (time_conv2d[i], mAP50_conv2d), textcoords="offset points",
                xytext=label_offsets[i], fontsize=8.5, color="#555555")
ax.set_xscale("log")
ax.set_xlim(0.1, 30)
ax.set_ylim(0.16, 0.235)
ax.set_xlabel("Avg. inference time per image, seconds (log scale)")
ax.set_ylabel("mAP50 (COCO2017 val)")
ax.set_title("Accuracy vs. latency trade-off\n(marker shape = device, color = convolution type)")
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CONV2D, markersize=10, label="YOLOv8-Conv2D"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SEP, markersize=10, label="YOLOv8-Separable"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#999999", markersize=8, label="GPU"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#999999", markersize=8, label="CPU"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#999999", markersize=8, label="Raspberry Pi 4"),
]
ax.legend(handles=legend_elems, frameon=False, loc="lower left", fontsize=9)
ax.grid(color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "map_vs_latency_tradeoff.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5: Real training convergence of YOLOv8-Separable (10 epochs, taken
# from the notebook's own saved training log — genuine run, not simulated)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_loss, marker="o", color=COLOR_SEP, label="train loss")
ax.plot(epochs, val_loss, marker="o", color=COLOR_CONV2D, label="val loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (box + class)")
ax.set_title("YOLOv8-Separable training convergence\n(real training log, 10 epochs, COCO2017)")
ax.set_xticks(epochs)
ax.legend(frameon=False)
ax.grid(color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
# annotate the epoch-1 warmup spike explicitly since it dwarfs the rest
ax.annotate("epoch 1 warm-up\n(loss=21.65)", (1, train_loss[0]), textcoords="offset points",
            xytext=(15, 0), fontsize=8.5, color="#555555")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "training_convergence_separable.png"), dpi=160)
plt.close(fig)

print("Saved figures to", FIG_DIR)
for f in sorted(os.listdir(FIG_DIR)):
    print(" -", f)
