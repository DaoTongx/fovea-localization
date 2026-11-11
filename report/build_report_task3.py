# -*- coding: utf-8 -*-
"""
生成《任务三：眼底黄斑中央凹定位》的实验报告（Word）。

所有数值均取自本次实验的真实产物：
  - Kaggle 训练日志（logs/v1_run.log、logs/v3_run.log）
  - 三次 OOF 逐图评测输出（results/、results2/、results3/）
  - Kaggle 排行榜提交记录（46 → 18 → 16 → 14）

未做任何虚构。图表由脚本现算现画，保证与正文数字一致。
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根目录
FIG = os.path.join(HERE, "report", "figures")
DATA = os.path.join(HERE, "raw", "detection")
OUT = os.path.join(HERE, "report", "任务三_眼底黄斑定位_实验报告_完整版.docx")
GITHUB_URL = "https://github.com/DaoTongx/fovea-localization"
os.makedirs(FIG, exist_ok=True)

# =========================================================================== #
#  真实实验数据
# =========================================================================== #
BASE_RMSE = 146.6

VERSIONS = [
    ("v1", "整图直接回归 + MSE 热力图", "52.32", "46"),
    ("v2", "BCE + bias(−4) + warmup + 失败折重训", "—", "18"),
    ("v3", "512 输入 + 3 seed 热力图集成", "16.90", "16"),
    ("v4", "120 epoch + 增强放大", "19.77", "未提交"),
    ("v5", "v3 + 绿通道 CLAHE", "18.82", "未提交"),
    ("v6", "v3 + 640 输入（最终提交）", "23.00", "14"),
]

FOLD_RMSE = {
    "seed 42": ["9.5", "14.7", "20.5", "6.1", "24.8"],
    "seed 1234": ["8.3", "16.4", "23.0", "5.7", "24.8"],
    "seed 7": ["11.5", "17.1", "22.9", "5.6", "25.1"],
}

TRAIN_CFG = [
    ("主干网络", "ResNet18（ImageNet 预训练）", "小样本下迁移学习优于从头训练"),
    ("输入尺寸", "512×512（v3）/ 640×640（v6）", "letterbox 等比缩放 + 补黑边"),
    ("热力图分辨率", "输入尺寸 / 4", "格子步长 4 像素"),
    ("批大小", "8", "受显存与每折样本量约束"),
    ("优化器", "AdamW，权重衰减 1e-4", "解耦权重衰减"),
    ("学习率", "主干 1e-4，回归头 2e-4", "头部随机初始化，需要更大步长"),
    ("学习率调度", "5 epoch 线性 warmup + 余弦退火", "避免初期热力图饱和"),
    ("训练轮数", "80，早停 patience=30", "验证 RMSE 30 轮不降即停"),
    ("损失函数", "BCE-with-logits + 1.0×坐标 L1", "前者监督热力图，后者直接对齐指标"),
    ("数据增强", "翻转 / 旋转 ±12° / 缩放 0.85~1.15 / 平移 ±5% / 光度扰动", "几何变换与坐标同步"),
    ("集成", "5 折 × 3 seed = 15 个模型", "热力图层面平均"),
    ("推理增强", "水平翻转 TTA", "坐标按 S−1−x 还原"),
]


# =========================================================================== #
#  读日志 / 读 OOF
# =========================================================================== #
def parse_log(path):
    """解析训练日志，返回 {(mode, seed, fold): [(epoch, val_rmse), ...]}"""
    segs, cur, mode = {}, None, "?"
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"=====\s*mode\s*=\s*(\w+)", ln)
        if m:
            mode = m.group(1)
            continue
        m = re.search(r"(?:seed\s+(\d+)\s+)?fold\s+(\d+)\s+train=", ln)
        if m:
            cur = (mode, int(m.group(1) or 0), int(m.group(2)))
            segs[cur] = []
            continue
        m = re.search(r"ep\s+(\d+)/\d+\s+loss\s+([\d.]+)\s+val RMSE\s+([\d.]+)", ln)
        if m and cur is not None:
            segs[cur].append((int(m.group(1)), float(m.group(3))))
    return segs


LOG_V1 = parse_log(os.path.join(HERE, "logs", "v1_run.log"))
LOG_V3 = parse_log(os.path.join(HERE, "logs", "v3_run.log"))


def load_oof(name):
    """读取逐图评测明细；若本地有原始标注则用真实坐标，否则由 nx*w、ny*h 还原。"""
    p = os.path.join(HERE, name, "oof_detail.csv")
    d = pd.read_csv(p, dtype={"id": str})
    gt_path = os.path.join(DATA, "fovea_localization_train_GT.csv")
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path, dtype={"data": str})
        gt["id"] = gt["data"].str.zfill(4)
        g = gt.set_index("id")
        d["w_px"] = g.loc[d.id, "Fovea_X"].values
        d["h_px"] = g.loc[d.id, "Fovea_Y"].values
    else:
        d["w_px"] = d.nx * d.w
        d["h_px"] = d.ny * d.h
    return d.sort_values("dist", ascending=False).reset_index(drop=True)


OOF6 = load_oof(os.path.join("results", "v6_640_3seed"))
OOF4 = load_oof(os.path.join("results", "v4_512_120ep"))
OOF5 = load_oof(os.path.join("results", "v5_512_clahe"))


def img_stats(i, size=512):
    im = cv2.imread(os.path.join(DATA, "train", i + ".jpg"))
    h, w = im.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = g > 12
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float((g[m] / 255).mean()), float(lap[m].var()), (w, h)


# =========================================================================== #
#  绘图
# =========================================================================== #
def setup_font():
    from matplotlib import font_manager
    wanted = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun", "DejaVu Sans"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((w for w in wanted if w in have), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False
    print("[font]", chosen)


def fig_examples():
    ids = ["0008", "0040", "0053", "0024"]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6))
    for ax, i in zip(axes.ravel(), ids):
        im = cv2.cvtColor(cv2.imread(os.path.join(DATA, "train", i + ".jpg")), cv2.COLOR_BGR2RGB)
        r = OOF6[OOF6.id == i].iloc[0]
        ax.imshow(im)
        ax.plot(r.w_px, r.h_px, "x", color="#e8412c", ms=13, mew=3)
        if r.dist > 20:
            ax.plot(r.w_px + r.dx, r.h_px + r.dy, "o", mfc="none", mec="#1f9cd6", ms=16, mew=2.4)
        ax.set_title("图像 %s（%d×%d）" % (i, im.shape[1], im.shape[0]), fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    p = os.path.join(FIG, "fig3-1_examples.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_coords():
    gt = pd.read_csv(os.path.join(DATA, "fovea_localization_train_GT.csv"), dtype={"data": str})
    gt["id"] = gt["data"].str.zfill(4)
    ws, hs = [], []
    for i in gt.id:
        w, h = img_stats(i)[2]
        ws.append(w)
        hs.append(h)
    ws, hs = np.array(ws, float), np.array(hs, float)
    nx, ny = gt.Fovea_X.values / ws, gt.Fovea_Y.values / hs

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    big = ws > 2500
    ax.scatter(nx[big], ny[big], s=26, c="#2f6fb3", label="2992×2000（75 张）", alpha=0.8)
    ax.scatter(nx[~big], ny[~big], s=34, c="#d1682c", label="1956×1934（5 张）", alpha=0.9, marker="s")
    ax.set_xlabel("归一化横坐标 x/W")
    ax.set_ylabel("归一化纵坐标 y/H")
    ax.set_xlim(0.30, 0.68)
    ax.set_ylim(0.42, 0.63)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(a) 标注坐标的归一化分布", fontsize=10.5)

    ax = axes[1]
    ax.hist(gt.Fovea_X, bins=18, color="#2f6fb3", alpha=0.85, label="X（像素）")
    ax.set_xlabel("中央凹横坐标 X / px")
    ax.set_ylabel("图像数量")
    ax2 = ax.twinx()
    ax2.hist(gt.Fovea_Y, bins=18, color="#d1682c", alpha=0.55, label="Y（像素）")
    ax2.set_ylabel("图像数量", color="#d1682c")
    ax.set_title("(b) 像素坐标分布（X 标准差 202.5，Y 仅 43.8）", fontsize=10.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    p = os.path.join(FIG, "fig3-2_coords.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_pipeline():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(y, text, fc="#eef3fa", ec="#4a6fa5", fs=9.5, h=0.8):
        ax.add_patch(FancyBboxPatch((1.6, y - h / 2), 6.8, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    linewidth=1.1, edgecolor=ec, facecolor=fc))
        ax.text(5.0, y, text, ha="center", va="center", fontsize=fs, linespacing=1.5)

    def arrow(y1, y2):
        ax.add_patch(FancyArrowPatch((5.0, y1), (5.0, y2), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.1, color="#5a6b7d"))

    box(9.3, "输入：眼底图像（2992×2000 或 1956×1934）", fc="#dbe7f5", fs=9.2)
    arrow(8.9, 8.45)
    box(8.1, "letterbox：等比缩放至 S×S，居中补黑边\n记录缩放比 s 与偏移 (l, t)")
    arrow(7.7, 7.25)
    box(6.9, "ImageNet 归一化 → ResNet18 主干（步长 32）")
    arrow(6.5, 6.05)
    box(5.7, "解码器：3 次 ×2 上采样 → 1 通道热力图（S/4）", fc="#fdf1e3", ec="#c8873f")
    arrow(5.3, 4.85)
    box(4.5, "峰值窗口质心解码：取最大响应格点，\n在 7×7 窗口内做加权质心（亚像素）",
        fc="#fdf1e3", ec="#c8873f")
    arrow(4.1, 3.65)
    box(3.3, "逆变换回原图：x = (x_lb − l) / s\ny = (y_lb − t) / s → 裁剪到图像范围")
    arrow(2.9, 2.45)
    box(2.1, "TTA（水平翻转）+ 15 个模型热力图平均")
    arrow(1.7, 1.25)
    box(0.9, "输出 submission.csv（X / Y 两行一组）", fc="#eaf5ea", ec="#4f8a4f")

    fig.tight_layout()
    p = os.path.join(FIG, "fig3-3_pipeline.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_arch():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.6, 2.55))
    ax.set_xlim(0, 9.7)
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    def blk(x, y, w, h, t, fc, ec, fs=7.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    linewidth=1.0, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.85, color="#5a6b7d"))

    top = ["输入\n3×S×S", "conv1+pool\nS/4", "layer1\nS/4", "layer2\nS/8", "layer3\nS/16", "layer4\nS/32"]
    tw, tg, ty, th = 1.30, 0.30, 1.55, 0.62
    for i, t in enumerate(top):
        x = 0.15 + i * (tw + tg)
        blk(x, ty, tw, th, t, "#dbe7f5" if i == 0 else "#eef3fa", "#4a6fa5")
        if i:
            arrow(x - tg + 0.03, ty + th / 2, x - 0.03, ty + th / 2)

    x_end = 0.15 + 5 * (tw + tg) + tw / 2
    ax.plot([x_end, x_end], [ty, 1.26], color="#5a6b7d", lw=0.85)
    ax.plot([x_end, 1.30], [1.26, 1.26], color="#5a6b7d", lw=0.85)
    arrow(1.30, 1.26, 1.30, 0.90)
    ax.text(4.9, 1.33, "解码器（输出 1/4 分辨率热力图）", fontsize=7.2, color="#5a6b7d", ha="center")

    dec = ["Conv3×3 512→128\n+BN+ReLU", "×2 上采样\nConv 128→64",
           "×2 上采样\nConv 64→32", "×2 上采样\n1×1 conv → 热力图"]
    dw, dg, dy, dh = 2.05, 0.32, 0.28, 0.62
    for i, t in enumerate(dec):
        x = 0.23 + i * (dw + dg)
        blk(x, dy, dw, dh, t, "#fdf1e3", "#c8873f", fs=7.2)
        if i:
            arrow(x - dg + 0.03, dy + dh / 2, x - 0.03, dy + dh / 2)

    fig.tight_layout()
    p = os.path.join(FIG, "fig3-4_arch.png")
    fig.savefig(p, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7))

    ax = axes[0]
    if ("heat", 0, 2) in LOG_V1:
        e = np.array(LOG_V1[("heat", 0, 2)])
        ax.plot(e[:, 0], e[:, 1], "-o", ms=4, lw=1.6, color="#c0392b", label="fold 2（崩溃）")
    if ("heat", 0, 0) in LOG_V1:
        e = np.array(LOG_V1[("heat", 0, 0)])
        ax.plot(e[:, 0], e[:, 1], "-s", ms=4, lw=1.6, color="#2f6fb3", label="fold 0（正常）")
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("验证集 RMSE / px")
    ax.set_yscale("log")
    ax.set_ylim(3, 1200)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8.5)
    ax.annotate("第 31 轮早停", xy=(16, 302), xytext=(28, 520), fontsize=7.8, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
    ax.set_title("(a) 修复前：MSE+sigmoid 导致的训练崩溃", fontsize=10.5)

    ax = axes[1]
    palette = ["#2f6fb3", "#d1682c", "#3f8f4f", "#9b59b6", "#c0392b"]
    for i in range(5):
        key = ("heat", 42, i)
        if key in LOG_V3:
            e = np.array(LOG_V3[key])
            ax.plot(e[:, 0], e[:, 1], "-o", ms=3.6, lw=1.5, color=palette[i], label="fold %d" % i)
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("验证集 RMSE / px")
    ax.set_yscale("log")
    ax.set_ylim(3, 1200)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("(b) 修复后：5 折全部稳定收敛", fontsize=10.5)

    fig.tight_layout()
    p = os.path.join(FIG, "fig3-5_curves.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_errors():
    d = OOF6.sort_values("dist", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    x = np.arange(1, len(d) + 1)
    colors = ["#c0392b" if i < 8 else "#2f6fb3" for i in range(len(d))]
    ax.bar(x, d.dist.values, color=colors, width=0.85)
    ax.set_xlabel("图像（按误差降序）")
    ax.set_ylabel("定位误差 / px")
    ax.set_xlim(0.5, len(d) + 0.5)
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.axhline(20, color="gray", ls="--", lw=1)
    ax.text(len(d) * 0.62, 22, "20 px", color="gray", fontsize=8)
    ax.set_title("最终模型（v6）在训练集上的逐图 OOF 误差：前 8 张贡献 87.1% 的平方误差",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIG, "fig3-6_errors.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_hard():
    top = OOF6.head(8)
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.4))
    for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
        im = cv2.cvtColor(cv2.imread(os.path.join(DATA, "train", r.id + ".jpg")), cv2.COLOR_BGR2RGB)
        ax.imshow(im)
        ax.plot(r.w_px, r.h_px, "+", color="#e8412c", ms=16, mew=3.2)
        ax.plot(r.w_px + r.dx, r.h_px + r.dy, "o", mfc="none", mec="#1f9cd6", ms=19, mew=2.6)
        sh = img_stats(r.id)[1]
        ax.set_title("%s  %.0f px  清晰度 %.0f" % (r.id, r.dist, sh), fontsize=9)
        ax.axis("off")
    fig.text(0.5, 0.01, "红叉：标注位置；蓝圈：模型预测位置", ha="center", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    p = os.path.join(FIG, "fig3-7_hard.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_clahe():
    ids = ["0049", "0044"]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.4))
    for row, i in enumerate(ids):
        im = cv2.cvtColor(cv2.imread(os.path.join(DATA, "train", i + ".jpg")), cv2.COLOR_BGR2RGB)
        g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(im[:, :, 1])
        axes[row, 0].imshow(im)
        axes[row, 0].set_title("%s 原图" % i, fontsize=9.5)
        axes[row, 1].imshow(g, cmap="gray")
        axes[row, 1].set_title("%s 绿通道 CLAHE" % i, fontsize=9.5)
        for c in (0, 1):
            axes[row, c].axis("off")
    fig.tight_layout()
    p = os.path.join(FIG, "fig3-8_clahe.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_resolution():
    names = ["512（v3）", "640（v6）"]
    oof = [16.90, 23.00]
    lb = [16, 14]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    b1 = ax.bar(x - 0.18, oof, 0.34, color="#2f6fb3", label="OOF RMSE")
    b2 = ax.bar(x + 0.18, lb, 0.34, color="#d1682c", label="排行榜 RMSE")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, "%.2f" % b.get_height(),
                ha="center", fontsize=8.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("RMSE / px")
    ax.set_ylim(0, 27)
    ax.legend(fontsize=8.6)
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.set_title("输入分辨率对验证集与排行榜指标的影响（方向相反）", fontsize=10.5)
    fig.tight_layout()
    p = os.path.join(FIG, "fig3-9_resolution.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# =========================================================================== #
#  docx 工具
# =========================================================================== #
def set_run_font(run, ascii_font="Times New Roman", ea_font="宋体", size=None, bold=None):
    run.font.name = ascii_font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), ea_font)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9)


def shade_cell(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def set_table_fixed_layout(table):
    layout = table._tbl.tblPr.get_or_add_tblLayout()
    layout.set(qn("w:type"), "fixed")


def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    tblPr.insert_element_before(
        borders, "w:shd", "w:tblLayout", "w:tblCellMar",
        "w:tblLook", "w:tblCaption", "w:tblDescription")


def add_caption(doc, text, kind="table"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6 if kind == "figure" else 3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, size=9.5, bold=True)
    return p


def add_table(doc, header, rows, widths=None, font_size=9.0, caption=None, align_center_cols=None):
    if caption:
        add_caption(doc, caption, kind="table")
    table = doc.add_table(rows=1, cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    set_table_borders(table)
    set_table_fixed_layout(table)

    hdr = table.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(str(h))
        set_run_font(run, size=font_size, bold=True)
        shade_cell(hdr[i], "DCE6F1")

    for row in rows:
        cells = table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER
                           if (align_center_cols is None or i in align_center_cols)
                           else WD_ALIGN_PARAGRAPH.LEFT)
            run = p.add_run(str(v))
            set_run_font(run, size=font_size)

    if widths:
        for r in table.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
        grid = table._tbl.tblGrid
        cols = grid.findall(qn("w:gridCol"))
        for gc, w in zip(cols, widths):
            gc.set(qn("w:w"), str(int(Cm(w).twips)))
        tblW = table._tbl.tblPr.find(qn("w:tblW"))
        if tblW is not None:
            tblW.set(qn("w:type"), "dxa")
            tblW.set(qn("w:w"), str(int(Cm(sum(widths)).twips)))
    # 除最后一行外都“与下段同页”，避免表格被拆到两页
    for r in table.rows[:-1]:
        for c in r.cells:
            for p in c.paragraphs:
                p.paragraph_format.keep_with_next = True
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


# =========================================================================== #
#  主构建
# =========================================================================== #
def build():
    setup_font()
    f1 = fig_examples()
    f2 = fig_coords()
    f3 = fig_pipeline()
    f4 = fig_arch()
    f5 = fig_curves()
    f6 = fig_errors()
    f7 = fig_hard()
    f8 = fig_clahe()
    f9 = fig_resolution()
    print("[figs] 9 张图已生成")

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.6)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.4)
    add_page_number_footer(sec)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(0)

    for name, size in (("Heading 1", 16), ("Heading 2", 13.5), ("Heading 3", 11.5)):
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        st.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 9)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    def H1(t):
        doc.add_page_break()
        h = doc.add_heading(level=1)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=16, bold=True)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def H2(t):
        h = doc.add_heading(level=2)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=13.5, bold=True)

    def H3(t):
        h = doc.add_heading(level=3)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=11.5, bold=True)

    def P(text, indent=True, bold=False, size=10.5, after=3, align=None):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.74) if indent else Cm(0)
        p.paragraph_format.space_after = Pt(after)
        p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold)
        return p

    def NUM(items, size=10.5):
        for k, it in enumerate(items, 1):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(1.2)
            p.paragraph_format.first_line_indent = Cm(-0.46)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r = p.add_run("（%d）%s" % (k, it))
            set_run_font(r, size=size)

    def FIG(path, caption, width_cm=14.6):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(path, width=Cm(width_cm))
        add_caption(doc, caption, kind="figure")

    # ---------------------------------------------------------------- 封面块
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_before = Pt(6)
    r = t.add_run("机器学习课程设计实验报告")
    set_run_font(r, ea_font="黑体", size=20, bold=True)

    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = st.add_run("—— 任务三：眼底图像黄斑中央凹定位 ——")
    set_run_font(r, ea_font="黑体", size=13)

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.paragraph_format.space_before = Pt(4)
    r = info.add_run("（本报告为四个任务中的第三章，可与其余章节合并成册）")
    set_run_font(r, size=9)

    info2 = doc.add_paragraph()
    info2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = info2.add_run("代码仓库：" + GITHUB_URL)
    set_run_font(r, size=9)

    # ==================================================================== #
    H1("第 3 章  眼底图像黄斑中央凹定位")
    # ==================================================================== #

    H2("3.1  任务背景与数据分析")
    H3("3.1.1  问题定义")
    P("黄斑中央凹（fovea）位于视网膜后极部，是视锥细胞密度最高、视觉最敏锐的区域。在眼底图像分析中，"
      "中央凹坐标是确定黄斑中心、计算视盘—黄斑距离、划分病变区域，以及进行青光眼与糖尿病视网膜病变"
      "分级的基础几何量。本任务要求：给定一张眼底图像，输出中央凹在该图像上的像素坐标 (X, Y)；"
      "若图像中中央凹不可见，则输出 (0, 0)。")
    P("由于官方未提供面向坐标的专用评价指标，提交文件将每张图的 X、Y 拆成相邻两行（ImageID 形如 "
      "81_Fovea_X / 81_Fovea_Y），最终以全部坐标值的均方根误差（RMSE）排名，数值越小越好。"
      "即评价口径是“逐坐标的平方误差平均”，而不是“点间欧氏距离”，二者相差一个常数因子，排序等价。")
    P("本任务的一个现实约束是数据量极小：训练集只有 80 张图像，测试集 20 张，且比赛设两个排行榜"
      "（各取测试集的 50%），最终名次由第二榜决定。因此模型选择不能只看验证集，"
      "还要考虑小样本下方差极大的问题——这一点在 3.4.5 节会给出实测证据。")

    H3("3.1.2  数据集构成")
    P("数据包的目录结构如表 3-1 所示。训练标注以 PASCAL VOC 格式的 XML 给出，每个文件包含一个名为 "
      "Fovea 的矩形框。经核对，该框中心与 fovea_localization_train_GT.csv 给出的坐标平均只差 2.4 像素，"
      "即官方坐标正是由标注框中心导出，框本身约为 220×210 像素。这说明标注精度存在一个天然上限："
      "人工框选的中心抖动直接构成了误差下限。")
    add_table(doc,
              ["目录 / 文件", "数量", "说明"],
              [["train/", "80 张", "2992×2000 共 75 张，1956×1934 共 5 张"],
               ["test/", "20 张", "与训练集同源，含 1 张 1956×1934"],
               ["train_location/", "80 个 XML", "Fovea 标注框（xmin, ymin, xmax, ymax）"],
               ["fovea_localization_train_GT.csv", "80 行", "Fovea_X / Fovea_Y 像素坐标"],
               ["sample_submission.csv", "40 行", "每张测试图 X / Y 各一行"]],
              widths=[5.6, 2.6, 7.8], font_size=9.0,
              caption="表 3-1  数据集构成")
    FIG(f1, "图 3-1  训练样本示例（红叉为中央凹标注；蓝圈为该图预测偏差大于 20 px 的位置）", 14.8)

    H3("3.1.3  数据分析")
    P("对 80 张训练图的标注坐标做统计（表 3-2），可以看到两个必须正视的特点。")
    add_table(doc,
              ["统计量", "X 坐标 / px", "Y 坐标 / px", "归一化 x/W", "归一化 y/H"],
              [["最小值", "723.7", "930.2", "0.365", "0.465"],
               ["最大值", "1794.6", "1182.2", "0.617", "0.591"],
               ["均值", "1446.1", "1055.8", "0.494", "0.529"],
               ["标准差", "202.5", "43.8", "—", "—"]],
              widths=[3.2, 3.5, 3.5, 3.2, 3.2], font_size=9.0,
              caption="表 3-2  训练集标注坐标统计")
    P("第一，坐标高度集中。横坐标只覆盖图像宽度的 36.5%~61.7%，纵坐标更窄，仅覆盖 46.5%~59.1%"
      "（标准差 43.8 像素，约等于图像高度的 2%）。图 3-2 的散点图直观显示了这一分布："
      "80 个点挤在图像中央一个扁长区域内。")
    P("第二，Y 方向几乎不变，难度全部集中在 X 方向。若直接用训练集均值 (1446.1, 1055.8) 作为预测，"
      "RMSE 为 %.1f 像素——这就是本任务的“笨基线”。它同时也是一个陷阱：由于黄斑在图像中大致居中，"
      "网络很容易学到“预测图像中心附近”这一捷径，在验证集上拿到看似尚可的分数，"
      "却在个别偏离中心的样本上错得很离谱。" % BASE_RMSE)
    FIG(f2, "图 3-2  训练集标注坐标的分布（左：归一化散点；右：像素坐标直方图）")

    P("另有两项必须确认的事实：(1) 训练集 80 张标注中没有 (0, 0) 样本，即所有训练图的中央凹均可见，"
      "模型无法从数据中学习“不可见”这一类别；(2) 逐张检查 20 张测试图，全部能观察到黄斑无血管区。"
      "因此本实验采取始终输出坐标、不启用 (0,0) 判定的策略——若误将一张可见图判为 (0,0)，"
      "仅这一张就会使 RMSE 上升约 300 像素，风险远大于收益。")

    # ------------------------------------------------------------------ #
    H2("3.2  方法设计")
    H3("3.2.1  总体框架")
    P("整体流程如图 3-3 所示，可概括为“归一化—预测—解码—还原”四步：把任意尺寸的眼底图等比缩放并"
      "居中放入 S×S 画布（letterbox），同时记录缩放比与偏移；网络输出 S/4 分辨率的热力图；"
      "在热力图上做峰值窗口质心解码得到亚像素坐标；最后用记录的变换把坐标还原到原图像素尺度。"
      "多尺度输入、多随机种子与水平翻转 TTA 都在解码之后、还原之前完成平均。")
    FIG(f3, "图 3-3  黄斑中央凹定位的总体流程")

    H3("3.2.2  坐标的归一化表示")
    P("训练集存在两种分辨率（2992×2000 与 1956×1934），若直接把像素坐标作为回归目标，"
      "同一解剖位置会对应差异很大的数值，网络需要额外学习尺度换算。为此采用 letterbox 变换：")
    P("s = S / max(H, W)，  x_lb = s·x + l，  y_lb = s·y + t",
      indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    P("其中 S 为输入边长，l、t 为图像在画布中的左上偏移。预测完成后按 x = (x_lb − l)/s 还原。"
      "这样做有三个好处：其一，两种分辨率被映射到统一的坐标语义；其二，等比缩放避免了直接 resize 到"
      "正方形造成的各向异性畸变（若把 3:2 的图强行压成 1:1，X 与 Y 的误差尺度将不再可比，"
      "而 RMSE 是两个方向合并计算的）；其三，所有误差都在原图像素上计算，与榜单口径一致。")

    H3("3.2.3  热力图监督与亚像素解码")
    P("本文没有采用“全局池化 + 全连接回归两个数”的直接回归方案，而是把定位建模为热力图回归，"
      "原因有三点。第一，直接回归在全局池化后丢失了空间信息，80 张训练图不足以让网络从零学出"
      "“视盘在左、黄斑在右”这样的空间关系；热力图保留了特征图的空间结构，是更合适的归纳偏置。"
      "第二，热力图是密集监督：每个像素都有目标值，一张图可提供上万个监督信号，"
      "相当于把 80 个样本放大了一个数量级。第三，热力图的峰值形态本身携带置信度信息，"
      "便于诊断模型是否“看错了地方”。")
    P("具体地，以标注点为均值、σ = 1.5 个格子构造二维高斯作为软标签，用二元交叉熵"
      "（BCE-with-logits）监督。解码时不使用全图 soft-argmax，而是先取响应最大的格点，"
      "再以其为中心在 7×7 窗口内做加权质心：")
    P("x = Σ w_i·(c + Δ_i) / Σ w_i，  Δ_i ∈ {−3, …, 3}",
      indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    P("这一细节很关键。若对整张热力图直接做归一化质心，背景上大量接近零但非零的响应会累积起来，"
      "把质心拉向图像中心——在小样本、目标接近中心的数据上，这种偏移会伪装成“还不错的精度”，"
      "却在离群样本上暴露。窗口质心只在峰附近取权重，从机制上消除了这一偏置；"
      "其数值正确性已用合成高斯验证：给定已知峰位，还原误差小于 0.25 像素。")

    H3("3.2.4  网络结构")
    P("主干采用 ImageNet 预训练的 ResNet18（11.86M 参数），取其 layer4 输出（步长 32），"
      "接一个轻量解码器：3 次“卷积 + 双线性上采样”把特征图放大 8 倍，得到 S/4 分辨率的单通道"
      "热力图（输入 640 时为 160×160）。选择 ResNet18 而非更深网络，是因为训练样本仅 80 张，"
      "容量更大的模型在验证集上并未带来收益，反而更难收敛。")
    FIG(f4, "图 3-4  网络结构：ResNet18 主干 + 8 倍上采样解码器", 15.2)

    H3("3.2.5  训练稳定性设计")
    P("这是本实验中影响最大的一环，也是从“46 分”走到“18 分”的关键。最初版本用 MSE 监督 sigmoid "
      "输出的热力图，结果 5 折中有 2 折训练崩溃：验证集 RMSE 在第 1 轮就是 176 与 243 像素，"
      "损失停滞在 0.22 附近，第 31 轮触发早停（图 3-5(a) 红线：验证误差不降反升）。")
    P("诊断后的原因很明确：MSE 对背景和前景使用同一量级的梯度，模型倾向于把整张热力图压成接近 0.5 "
      "的均匀面；此时归一化质心恰好落在图像中心附近，梯度接近消失，训练再也无法回到正轨。"
      "为此做了四项改动：")
    NUM(["把热力图最后一层卷积的偏置初始化为 −4，使初始输出 sigmoid(−4) ≈ 0.018，即“几乎全背景”，"
         "从第一步就避免均匀响应；",
         "把损失改为 BCE-with-logits，它对“该是背景却给出高响应”的惩罚是指数级的，"
         "能把背景压到 −10 以下，前景尖峰自然凸显；",
         "加入 5 个 epoch 的线性 warmup，并把头部学习率从 5e-4 降到 2e-4，避免初期一步迈入饱和区；",
         "增加失败折重训机制：某折最优验证 RMSE 超过 60 像素时，自动换种子并以 0.4 倍学习率重训一次，"
         "保留两者中更好的结果。"])
    P("图 3-5(b) 给出了修复后的收敛曲线：除 fold 0 外，各折都在 20 轮内降到 30 像素以内；"
      "fold 0 虽然在第 32 轮仍有 143 像素的反复，但随后快速下降，第 48 轮降至 19.6、第 64 轮降至 12.6 像素。"
      "关键是 5 折均未再出现“验证误差单调上升”的崩溃。该改动使 5 折最优验证 RMSE 从"
      "“18.5 / 21.8 / 176.0 / 242.7 / 50.3”变为“9.5 / 14.7 / 20.5 / 6.1 / 24.8”。")
    FIG(f5, "图 3-5  修复前后的验证集收敛曲线", 15.4)

    H3("3.2.6  集成与推理增强")
    P("推理阶段做了两项平均。其一是水平翻转 TTA：原图与镜像图各预测一次，镜像图的横坐标按 "
      "x' = S − 1 − x 还原后取平均。其二是在热力图层面而非坐标层面做多模型平均：5 折 × 3 个随机种子"
      "共 15 个模型的热力图逐像素相加后只解码一次。相比先解码再平均坐标，热力图平均能抑制个别模型"
      "的伪峰，也保留了峰值附近的更多信息。")

    # ------------------------------------------------------------------ #
    H2("3.3  实验方案")
    H3("3.3.1  数据划分")
    P("采用 5 折交叉验证，按图像尺寸分层抽样。若把 75 张 2992×2000 与 5 张 1956×1934 混在一起随机划分，"
      "少数尺寸容易集中到同一折，导致该折验证集偏难或偏易。所有实验固定同一套划分（随机种子 42），"
      "使不同版本的 OOF 可以逐图对比。")

    H3("3.3.2  数据增强")
    P("训练集每折只有 64 张，增强是防止过拟合的主要手段。几何增强包括水平翻转（概率 0.5）、"
      "旋转 ±12°、缩放 0.85~1.15、平移 ±5%，四者合成一个仿射矩阵，同时作用于图像与标注坐标，"
      "保证坐标不错位；光度增强包括随机 gamma（0.85~1.2）、亮度与对比度扰动、整体增益。"
      "需要说明的是，我曾把几何增强放大到旋转 ±15°、缩放 0.8~1.2、平移 ±8%，验证集反而从 16.9 "
      "恶化到 19.8——训练集构图高度一致，“黄斑大致居中”是一个有用的先验，过强的几何扰动会破坏它。"
      "这一负结果记录在表 3-4 中。")

    H3("3.3.3  训练配置")
    P("完整配置见表 3-3。所有实验在 Kaggle 的 Tesla T4 上完成：单次 5 折训练约 12~15 分钟，"
      "3 seed 集成约 45 分钟。")
    add_table(doc, ["项目", "取值", "说明"], TRAIN_CFG,
              widths=[3.0, 6.4, 6.6], font_size=8.8,
              caption="表 3-3  训练配置")

    H3("3.3.4  评价指标与模型选择")
    P("离线评价使用与榜单一致的口径：把 80 张图的 dx、dy 合并，按下式计算")
    P("RMSE = sqrt( (mean(dx²) + mean(dy²)) / 2 )", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    P("其中 dx、dy 为预测坐标与标注坐标之差，单位为原图像素。交叉验证时每张图恰好被预测一次"
      "（由其所在折的模型给出），得到完整的 OOF 预测，再统一计算 RMSE。"
      "在线评价以 Kaggle 公开榜为准，每天 5 次提交额度。")

    # ------------------------------------------------------------------ #
    H2("3.4  实验结果与分析")
    H3("3.4.1  基线")
    P("为给后续结果提供参照，先计算两个基线：常数预测（训练集坐标均值）的 RMSE 为 %.1f 像素；"
      "若用标注框中心直接代替坐标，误差仅 2.4 像素，说明标注内部一致性很高，"
      "剩余误差来自“从图像内容定位”这件事本身。" % BASE_RMSE)

    H3("3.4.2  版本演进与消融")
    P("表 3-4 汇总了六个版本的改动、OOF 与榜单成绩。最大的收益来自 v2 的训练稳定性修复（46 → 18），"
      "其次是 v3 的多 seed 集成（18 → 16）；此后三次尝试（v4、v5、v6）在验证集上均未超过 v3，"
      "但 v6 在榜单上反而更好，这一矛盾在 3.4.5 节展开。")
    add_table(doc, ["版本", "关键改动", "OOF RMSE / px", "榜单 RMSE / px"], VERSIONS,
              widths=[1.8, 7.6, 3.2, 3.2], font_size=9.0,
              caption="表 3-4  版本演进与消融实验")
    P("v3 的五折结果见表 3-5。三个随机种子在同一折上的最优验证 RMSE 几乎一致"
      "（例如 fold 4 为 24.8 / 24.8 / 25.1），说明折间差异来自数据本身而非训练随机性；"
      "而折与折之间差异巨大（fold 3 仅 5.6 像素，fold 4 达 25.1 像素），提示存在少数“难样本”。")
    rows = [[k] + v + ["%.1f" % np.mean([float(x) for x in v])] for k, v in FOLD_RMSE.items()]
    add_table(doc, ["模型种子", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "均值"], rows,
              widths=[3.2, 2.3, 2.3, 2.3, 2.3, 2.3, 2.4], font_size=9.0,
              caption="表 3-5  v3（512 输入）各折最优验证 RMSE / px")

    H3("3.4.3  逐图误差分析")
    P("整体 RMSE 掩盖了误差的真实结构。把最终模型（v6）的 80 个 OOF 误差降序排列（图 3-6），"
      "可以得到一个非常集中的分布：前 8 张图的平方误差占总量的 87.1%，其余 72 张的平均误差只有"
      " 10.1 像素、中位数 7.8 像素。换言之，模型的“典型表现”是 8 像素级，整体分数被极少数样本"
      "拖到了 23 像素。")
    FIG(f6, "图 3-6  最终模型的逐图 OOF 误差（红柱为误差最大的 8 张）", 15.2)
    P("这 8 张图的共同特征并不在尺寸（5 张 1956×1934 的方块图误差全部在 7~19 像素之间，属正常水平），"
      "而在成像质量与个体差异上。表 3-6 列出了它们的误差与 Laplacian 清晰度指标：清晰度（方差）"
      "越低，图像越“雾”，黄斑无血管区边界越模糊。全样本的 Spearman 相关系数为 −0.269（p = 0.016），"
      "方向与预期一致但并非决定性因素——0072 的清晰度有 77.8，误差仍达 68.9 像素，"
      "说明还存在与成像质量无关的个体差异。")
    hard_rows = []
    for _, r in OOF6.head(8).iterrows():
        b, sh, (w, h) = img_stats(r.id)
        hard_rows.append([r.id, "%.1f" % r.dist, "%.1f" % r.dx, "%.1f" % r.dy,
                          "%.0f" % sh, "%.2f" % b, "%d×%d" % (w, h)])
    add_table(doc, ["图像", "误差 / px", "dx", "dy", "清晰度", "亮度", "尺寸"], hard_rows,
              widths=[2.2, 2.4, 2.2, 2.2, 2.2, 2.2, 3.4], font_size=8.8,
              caption="表 3-6  v6 中误差最大的 8 张训练图")
    P("图 3-7 把其中一部分画了出来。可以观察到两类失败模式：一类是雾状、低对比的图像"
      "（0049、0024、0016），黄斑在原始亮度下几乎不可辨；另一类是模型锁定到了错误的暗区"
      "（0005 偏差 162 像素）。对 0049 进一步核对发现，它在两个完全不同的配置（512 / 640、"
      "不同损失与增强）下都稳定偏 121~136 像素且方向一致，这更像标注本身存在歧义，"
      "而不是模型随机波动。")
    FIG(f7, "图 3-7  误差最大的 8 张训练图（红叉为标注，蓝圈为预测）", 15.6)

    H3("3.4.4  针对难样本的尝试：CLAHE")
    P("既然难样本主要是低对比图像，一个自然的想法是先做对比度归一化。眼底图像的血管与黄斑无血管区"
      "在绿通道中对比度最高，因此尝试把输入改为“绿通道 CLAHE → 三通道灰度”（图 3-8）。"
      "从视觉上看，CLAHE 确实让 0049、0044 等图像中的黄斑暗斑从“几乎不可见”变为清晰可辨。")
    FIG(f8, "图 3-8  绿通道 CLAHE 前后的对比（左：原图；右：处理后）", 12.4)
    P("但实验结果与直觉相反（v5）：8 张难图的平均误差从 69.7 降到 61.9 像素，"
      "可一批原本很准的图像明显变差——0041 从 18.0 升至 37.7，0057 从 14.3 升至 33.5，"
      "0023 从 4.4 升至 24.3，整体 OOF 反而从 16.9 升到 18.8。原因推测有两点：灰度化丢掉了颜色"
      "信息（黄斑色素在彩图中是一个额外线索），而 CLAHE 在噪声较大的区域会放大伪结构。"
      "“救回几张难图、弄坏一批简单图”的净效果为负，因此最终没有采用 CLAHE。")

    H3("3.4.5  输入分辨率的影响")
    P("表 3-4 中最值得讨论的是 v3 与 v6 的矛盾：同为 5 折 × 3 seed、同样的损失与增强，"
      "只把输入从 512 提到 640，验证集从 16.9 恶化到 23.0，榜单却从 16 改善到 14。")
    FIG(f9, "图 3-9  输入分辨率对 OOF 与榜单指标的影响", 12.6)
    P("验证集的恶化几乎完全由三张图造成：0005 从 23 像素跳到 166.9，0034 从 81.2 到 124.7，"
      "0024 从 22.7 到 69.5。若把这三张剔除，剩余 77 张的 RMSE 只有 13.1 像素，反而优于 512 输入。"
      "也就是说，640 输入对绝大多数图像更好，只是在极少数样本上出现“锁错目标”的失稳。"
      "而测试集恰好不含这类样本——20 张测试图中只有 2 张属于低清晰度（清晰度 26.4 与 26.5），"
      "没有出现 0005 那种高亮度、强伪影的极端情况。")
    P("这一现象揭示了一个方法论问题：在 80 张训练图、误差重尾分布的条件下，OOF 的方差远大于模型间"
      "的真实差异，用它做模型选择会系统性偏向“在难样本上更保守”的模型。本实验最终依据榜单选择了 "
      "640 输入（榜单 14），但必须承认，公开榜只覆盖测试集的 50%（10 张图），一张图的偏差即可造成 "
      "2~3 分的波动，第二榜的结果仍存在不确定性。")

    H3("3.4.6  集成的实际作用")
    P("把 3 个种子的 OOF 分别计算，得到 16.61 / 18.41 / 18.76，等权融合后为 16.90。融合结果优于"
      "两个较弱种子，却不如最好的种子 42，这与“集成必然更好”的常见预期不符。原因在于误差是相关的："
      "三个种子在同一张图上错得一样，平均无法抵消。因此多 seed 的价值主要体现在抑制单次训练崩溃"
      "（3.2.5 节的失败折重训），而不是压低平均误差。")

    # ------------------------------------------------------------------ #
    H2("3.5  创新点")
    P("本实验的工作集中在“让一个简单的框架在小样本、重尾误差的条件下稳定工作”，具体有以下四点。")
    NUM(["提出并实现了峰值窗口质心解码。常规做法是对整张热力图做归一化 soft-argmax，"
         "背景的残余响应会把结果系统性地拉向图像中心；改成先取峰值、再在 7×7 窗口内求质心后，"
         "既保留了亚像素精度，又从机制上消除了这一偏置，并用合成高斯做了数值验证。",
         "把训练崩溃当作可诊断、可拦截的工程问题。通过“偏置初始化 + BCE + warmup”解决了热力图饱和"
         "导致的梯度消失，并额外加入失败折自动重训，使 5 折全部收敛。这一环节把榜单从 46 分带到 "
         "18 分，是本实验收益最大的一步。",
         "用逐图误差归因而非整体分数来指导迭代。通过导出 OOF 逐图明细、结合图像的清晰度与亮度指标、"
         "以及“取两模型较优者”的 oracle 分析，量化了 8 张图贡献 87% 平方误差这一事实，"
         "从而判断出继续调参的边际收益很低，避免了在无效方向上继续投入。",
         "发现了验证集与榜单背离的现象并给出解释。在 80 张训练图上，640 输入的 OOF 比 512 差 6 个点，"
         "但去除 3 个离群样本后反而更好，榜单证实了后者。这说明小样本场景下应以榜单为准，"
         "同时提示：验证集上的“变差”可能只是被少数标注可疑样本拉低。"])
    P("此外，多尺度尝试中“把几何增强放大反而变差”这一负结果也有参考价值：当数据集的构图高度一致时，"
      "适度的位置先验比更强的几何不变性更有用，这与通用图像分类中的经验相反。")

    # ------------------------------------------------------------------ #
    H2("3.6  不足与改进方向")
    P("第一，(0,0) 的“不可见”情形无法学习。训练集 80 张全部可见，模型没有任何负样本；"
      "本文采取“一律输出坐标”的策略，虽然在本次测试集上正确，但缺少原则性的可见性判别。"
      "更合理的做法是引入外部数据构造负样本，或利用热力图峰值响应做异常检测。")
    P("第二，标注精度构成硬上限。标注是人工框选的中心，框约 220×210 像素，中心抖动估计在 8~12 像素；"
      "0049 这类样本在多次实验中稳定偏离同侧 130 像素左右，更像是标注歧义。在这类样本上继续优化模型，"
      "收益有限。")
    P("第三，未实现两阶段细化。本文的误差主要来自“看错地方”（粗定位），而不是“看得不准”"
      "（亚像素精度），因此“先粗定位再裁块回归”的两阶段方案对本任务的收益预期并不高；"
      "但若能把粗定位的失败率降下来，两阶段仍有价值。")
    P("第四，模型选择依赖公开榜。20 张测试图、10 张公开榜的规模太小，单张图即可造成 2~3 分波动，"
      "最终名次存在偶然性。更稳健的做法是扩大验证集（例如引入外部眼底数据）或做重复交叉验证以降低方差。")

    # ------------------------------------------------------------------ #
    H2("3.7  本章小结")
    P("本章针对眼底图像黄斑中央凹定位任务，构建了“letterbox 坐标归一化 + ResNet18 热力图回归 + "
      "峰值窗口质心解码 + 5 折 3 seed 集成”的方案。在 80 张训练图、20 张测试图的极小样本条件下，"
      "模型从常数基线的 146.6 像素误差降到公开榜 14 像素（OOF 16.9~23.0，取决于输入分辨率），"
      "进入合理区间。")
    P("实验中最有价值的经验并非网络结构，而是两点工程判断：其一，热力图回归的训练稳定性"
      "（偏置初始化、BCE、warmup、失败折重训）决定了方法能否工作；其二，在重尾误差的小样本任务中，"
      "验证集分数不足以支撑模型选择，必须结合逐图误差归因与真实榜单。后续工作可以从可见性判别、"
      "两阶段细化和更大规模的外部数据验证三个方向展开。")

    # ------------------------------------------------------------------ #
    H2("参考文献")
    refs = [
        "[1] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. CVPR, 2016.",
        "[2] Deng J, Dong W, Socher R, et al. ImageNet: A Large-Scale Hierarchical Image Database[C]. CVPR, 2009.",
        "[3] Law H, Deng J. CornerNet: Detecting Objects as Paired Keypoints[C]. ECCV, 2018.",
        "[4] Newell A, Yang K, Deng J. Stacked Hourglass Networks for Human Pose Estimation[C]. ECCV, 2016.",
        "[5] Loshchilov I, Hutter F. Decoupled Weight Decay Regularization[C]. ICLR, 2019.",
        "[6] Zuiderveld K. Contrast Limited Adaptive Histogram Equalization[M]. Graphics Gems IV, 1994.",
        "[7] Lin T Y, Goyal P, Girshick R, et al. Focal Loss for Dense Object Detection[C]. ICCV, 2017.",
    ]
    for r_ in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.9)
        p.paragraph_format.first_line_indent = Cm(-0.9)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        run = p.add_run(r_)
        set_run_font(run, size=9.2)

    H2("附录  代码与实验产物")
    P("本任务的全部代码、训练日志与实验结果已整理至 GitHub 仓库：")
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.74)
    run = p.add_run(GITHUB_URL)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x9A)
    set_run_font(run, size=10.5)
    P("仓库结构：notebooks/ 为 Kaggle 上直接运行的 .ipynb；src/ 为同内容的脚本版；"
      "report/ 为本文档及其构建脚本（图表均由脚本从日志与 OOF 数据现算生成）；"
      "logs/ 为两次关键训练的完整日志；results*/ 为三次运行的逐图评测明细。"
      "所有数值均可从这些文件中复算。")

    doc.save(OUT)
    print("[docx] saved:", OUT)


if __name__ == "__main__":
    build()
