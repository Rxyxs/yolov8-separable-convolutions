"""
Interactive Plotly chart built from the REAL thesis benchmark numbers
(same source as outputs/make_figures.py — README.md summary tables and
the saved notebook cell outputs). Self-contained HTML, no external JS.

Run: python outputs/make_interactive.py
"""
import os
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "interactive")
os.makedirs(OUT_DIR, exist_ok=True)

devices = ["GPU (RTX 4060 Ti)", "CPU (Ryzen 7 5700X)", "Raspberry Pi 4"]
time_conv2d = [0.1824, 0.4273, 13.9322]
time_sep = [0.1866, 0.4156, 1.4754]
fps_conv2d = [5.48, 2.34, 0.07]
fps_sep = [5.36, 2.41, 0.68]
mAP50_conv2d, mAP50_sep = 0.2119, 0.1751
params_conv2d, params_sep = 3_991_584, 1_258_715
size_conv2d_mb, size_sep_mb = 15.23, 4.80

COLOR_CONV2D = "#4C72B0"
COLOR_SEP = "#DD8452"

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=time_conv2d,
    y=[mAP50_conv2d] * 3,
    mode="markers+text",
    name="YOLOv8-Conv2D (standard)",
    text=devices,
    textposition="top center",
    marker=dict(size=18, color=COLOR_CONV2D, line=dict(width=1.5, color="white")),
    customdata=[[fps_conv2d[i], params_conv2d, size_conv2d_mb] for i in range(3)],
    hovertemplate=(
        "<b>YOLOv8-Conv2D</b> — %{text}<br>"
        "Avg. latency: %{x:.4f} s/image<br>"
        "FPS: %{customdata[0]:.2f}<br>"
        "mAP50: %{y:.4f}<br>"
        "Params: %{customdata[1]:,} (%{customdata[2]:.2f} MB)"
        "<extra></extra>"
    ),
))

fig.add_trace(go.Scatter(
    x=time_sep,
    y=[mAP50_sep] * 3,
    mode="markers+text",
    name="YOLOv8-Separable (depthwise)",
    text=devices,
    textposition="bottom center",
    marker=dict(size=18, color=COLOR_SEP, line=dict(width=1.5, color="white")),
    customdata=[[fps_sep[i], params_sep, size_sep_mb] for i in range(3)],
    hovertemplate=(
        "<b>YOLOv8-Separable</b> — %{text}<br>"
        "Avg. latency: %{x:.4f} s/image<br>"
        "FPS: %{customdata[0]:.2f}<br>"
        "mAP50: %{y:.4f}<br>"
        "Params: %{customdata[1]:,} (%{customdata[2]:.2f} MB)"
        "<extra></extra>"
    ),
))

fig.update_layout(
    title=dict(
        text=(
            "YOLOv8 — Accuracy vs. Latency Trade-off Across Devices<br>"
            "<sup>Real thesis benchmark: COCO2017 val, 640×640 — "
            "Universidad Mayor, Chile (2026)</sup>"
        ),
    ),
    xaxis=dict(title="Avg. inference time per image (seconds, log scale)", type="log"),
    yaxis=dict(title="mAP50 (COCO2017 val)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
    template="plotly_white",
    font=dict(family="Arial, sans-serif", size=13, color="#333333"),
    width=980,
    height=620,
    margin=dict(t=110),
)

out_path = os.path.join(OUT_DIR, "latency_map_tradeoff.html")
fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
print("Saved", out_path)
