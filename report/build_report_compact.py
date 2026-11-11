# -*- coding: utf-8 -*-
"""精简版（6~8 页）《任务三：眼底黄斑中央凹定位》实验报告。

复用 build_report_task3.py 中的绘图与 docx 工具函数，只保留 4 张图与 3 张表。
"""

import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from build_report_task3 import (
    BASE_RMSE, FIG, GITHUB_URL, HERE, VERSIONS,
    add_page_number_footer, add_table, fig_arch, fig_curves, fig_errors, fig_pipeline,
    set_run_font, setup_font,
)

OUT = os.path.join(HERE, "report", "任务三_眼底黄斑定位_实验报告.docx")

# 精简后的版本表：只保留关键信息
VER_ROWS = [
    ("v1", "直接回归 + MSE 热力图（训练崩溃）", "52.32", "46"),
    ("v2", "BCE + bias(−4) + warmup + 失败折重训", "—", "18"),
    ("v3", "512 输入 + 3 seed 集成", "16.90", "16"),
    ("v4", "120 epoch + 增强放大", "19.77", "未提交"),
    ("v5", "v3 + 绿通道 CLAHE", "18.82", "未提交"),
    ("v6", "v3 + 640 输入（最终提交）", "23.00", "14"),
]

CFG_ROWS = [
    ("主干网络", "ResNet18（ImageNet 预训练，11.86M 参数）"),
    ("输入 / 输出", "512×512（v3）或 640×640（v6），letterbox 补黑边；1/4 分辨率热力图"),
    ("优化设置", "AdamW（wd=1e-4，batch 8），主干 1e-4、回归头 2e-4，5 epoch warmup + 余弦退火"),
    ("损失 / 轮数", "BCE-with-logits + 1.0×坐标 L1；80 epoch，早停 patience=30"),
    ("数据增强", "翻转、旋转 ±12°、缩放 0.85~1.15、平移 ±5°、光度扰动"),
    ("集成", "5 折 × 3 seed = 15 个模型，热力图层面平均 + 水平翻转 TTA"),
]


def build():
    setup_font()
    f_pipe = fig_pipeline()
    f_arch = fig_arch()
    f_curve = fig_curves()
    f_err = fig_errors()

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

    for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        st.paragraph_format.space_before = Pt(8 if name == "Heading 1" else 6)
        st.paragraph_format.space_after = Pt(3)
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    def H1(t):
        h = doc.add_heading(level=1)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=15, bold=True)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def H2(t):
        h = doc.add_heading(level=2)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=12.5, bold=True)
        h.paragraph_format.keep_with_next = True

    def H3(t):
        h = doc.add_heading(level=3)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=11, bold=True)
        h.paragraph_format.keep_with_next = True

    def P(text, indent=True, size=10.5, after=2, align=None):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.74) if indent else Cm(0)
        p.paragraph_format.space_after = Pt(after)
        p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(text)
        set_run_font(r, size=size)
        return p

    def NUM(items, size=10.5):
        for k, it in enumerate(items, 1):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(1.2)
            p.paragraph_format.first_line_indent = Cm(-0.46)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r = p.add_run("（%d）%s" % (k, it))
            set_run_font(r, size=size)

    def FIG(path, caption, width_cm=13.2):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(path, width=Cm(width_cm))
        from build_report_task3 import add_caption
        add_caption(doc, caption, kind="figure")

    # ---------------- 标题 ----------------
    h = doc.add_heading(level=1)
    r = h.add_run("第 3 章  眼底图像黄斑中央凹定位")
    set_run_font(r, ea_font="黑体", size=15, bold=True)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("（机器学习课程设计 · 任务三；代码：" + GITHUB_URL + "）")
    set_run_font(r, size=8.5)

    # ---------------- 3.1 ----------------
    H2("3.1  任务与数据")
    H3("3.1.1  问题定义")
    P("黄斑中央凹是视网膜上视锥细胞最密集、视觉最敏锐的区域，其坐标是划分黄斑区、计算视盘—黄斑距离、"
      "以及进行青光眼与糖网分级的基础几何量。任务要求：给定一张眼底图像输出中央凹的像素坐标 (X, Y)，"
      "不可见时输出 (0, 0)；提交文件把 X、Y 拆成相邻两行，按全部坐标值的 RMSE 排名。"
      "训练集 80 张、测试集 20 张，两个榜单各取测试集的一半。样本量极小是本任务最本质的约束。")

    H3("3.1.2  数据概况")
    P("数据包含 train/ 80 张、test/ 20 张、train_location/ 80 个 VOC 格式标注框与两个 csv。"
      "核对发现标注框中心与 GT 坐标平均只差 2.4 像素，说明 GT 由框中心导出，框约 220×210 像素，"
      "人工框选的抖动直接构成了误差下限。主要统计量见表 3-1。")
    add_table(doc,
              ["项目", "数值"],
              [["训练集 / 测试集", "80 张 / 20 张"],
               ["分辨率", "2992×2000（75 张）+ 1956×1934（5 张）"],
               ["X 范围 / 均值 / 标准差", "723.7~1794.6 px / 1446.1 / 202.5"],
               ["Y 范围 / 均值 / 标准差", "930.2~1182.2 px / 1055.8 / 43.8"],
               ["归一化范围", "x/W ∈ [0.365, 0.617]，y/H ∈ [0.465, 0.591]"],
               ["常数均值预测的 RMSE", "%.1f px" % BASE_RMSE]],
              widths=[5.4, 10.2], font_size=8.8, caption="表 3-1  数据与标注统计")
    P(f"两点观察决定了后续设计。其一，坐标高度集中且 Y 方向几乎不变（标准差仅 43.8 像素，约图高的 2%），"
      f"难度全部集中在 X 方向；用训练集均值直接预测即可得到 {BASE_RMSE:.1f} 像素的 RMSE，"
      f"这既是基线也是陷阱——网络很容易学到“预测图像中心”的捷径。其二，训练集标注中没有 (0, 0) 样本，"
      f"模型学不到“不可见”，而逐张检查 20 张测试图均可观察到黄斑，因此本文始终输出坐标，不启用 (0,0) 判定。")

    # ---------------- 3.2 ----------------
    H2("3.2  方法设计")
    H3("3.2.1  总体流程")
    P("整体流程如图 3-1：等比缩放并居中补黑边（letterbox）得到 S×S 输入，ResNet18 输出 1/4 分辨率热力图，"
      "在热力图上做峰值窗口质心解码得到亚像素坐标，再按记录的比例与偏移还原到原图像素，"
      "最后经水平翻转 TTA 与多模型热力图平均写入提交文件。")
    FIG(f_pipe, "图 3-1  黄斑中央凹定位的总体流程", 11.8)

    H3("3.2.2  坐标归一化")
    P("设 s = S/max(H,W)、左上偏移为 (l, t)，则 x_lb = s·x + l；预测后按 x = (x_lb − l)/s 还原。"
      "这样两种分辨率被映射到统一的坐标语义，等比缩放避免了直接压成正方形造成的各向异性畸变"
      "（若强行压成 1:1，X 与 Y 的误差尺度将不可比，而 RMSE 是两个方向合并计算的），"
      "且误差始终在原图像素上计算，与榜单口径一致。")

    H3("3.2.3  热力图监督与亚像素解码")
    P("本文把定位建模为热力图回归而非直接回归两个数，原因是：全局池化会丢失空间结构，80 张图不足以让"
      "网络从零学出“视盘在左、黄斑在右”的关系；热力图则保留了空间结构，且每个像素都有监督值，"
      "相当于把 80 个样本放大了一个数量级。训练时以标注点为中心、σ = 1.5 个格子构造高斯软标签，"
      "用 BCE-with-logits 监督。")
    P("解码时不使用全图 soft-argmax，而是先取响应最大的格点，再在以其为中心的 7×7 窗口内求加权质心。"
      "若对整图做归一化质心，背景上大量近零响应会累积起来把结果拉向图像中心——在小样本、目标居中"
      "的数据上这种偏移会伪装成“不错的精度”，却在离群样本上暴露。窗口质心只在峰附近取权重，"
      "从机制上消除了该偏置；合成高斯测试显示还原误差小于 0.25 像素。")

    H3("3.2.4  网络结构与训练稳定性")
    P("主干为 ImageNet 预训练的 ResNet18（11.86M 参数），取 layer4 输出（步长 32），接 3 次"
      "“卷积 + 双线性上采样”，得到 S/4 分辨率的单通道热力图（输入 640 时为 160×160），结构见图 3-2。")
    FIG(f_arch, "图 3-2  网络结构：ResNet18 主干 + 8 倍上采样解码器", 14.6)
    P("训练稳定性是本实验影响最大的一环。最初版本用 MSE 监督 sigmoid 输出，5 折中有 2 折崩溃："
      "验证 RMSE 首轮即为 176 与 243 像素，损失停滞在 0.22，第 31 轮早停。原因是 MSE 对背景与前景"
      "使用同一量级梯度，模型倾向于把热力图压成接近 0.5 的均匀面，此时归一化质心恰好落在图像中心，"
      "梯度接近消失。为此采取四项措施：")
    NUM(["最后一层卷积偏置初始化为 −4，使初始输出 sigmoid(−4) ≈ 0.018，即“几乎全背景”；",
         "损失改为 BCE-with-logits，对“该是背景却响应很高”施加指数级惩罚；",
         "加入 5 个 epoch 线性 warmup，头部学习率由 5e-4 降至 2e-4；",
         "某折最优验证 RMSE 超过 60 像素时自动换种子、以 0.4 倍学习率重训一次，保留更优者。"])
    P("修复后 5 折最优验证 RMSE 由“18.5 / 21.8 / 176.0 / 242.7 / 50.3”变为"
      "“9.5 / 14.7 / 20.5 / 6.1 / 24.8”，这一环节也是榜单从 46 分降到 18 分的主因。")

    H3("3.2.5  集成与推理增强")
    P("推理时做两项平均：水平翻转 TTA（镜像图横坐标按 S−1−x 还原）；5 折 × 3 个随机种子共 15 个模型"
      "的热力图逐像素相加后再统一解码。相比先解码再平均坐标，热力图平均能抑制个别模型的伪峰。")

    # ---------------- 3.3 ----------------
    H2("3.3  实验方案")
    P("采用 5 折交叉验证并按图像尺寸分层（75 张 2992×2000 与 5 张 1956×1934 混排随机划分时，"
      "少数尺寸容易集中到同一折）。几何增强为翻转、旋转 ±12°、缩放 0.85~1.15、平移 ±5°，"
      "四者合成同一仿射矩阵同时作用于图像与坐标；光度增强为 gamma、亮度、对比度与整体增益。"
      "曾把几何增强放大到旋转 ±15°、缩放 0.8~1.2、平移 ±8%，验证集反而由 16.9 恶化到 19.8——"
      "训练集构图高度一致，“黄斑大致居中”是有用先验，过强扰动会破坏它。主要配置见表 3-2。")
    add_table(doc, ["项目", "取值"], CFG_ROWS, widths=[3.6, 12.0], font_size=8.8,
              caption="表 3-2  训练配置")
    P("离线评价与榜单口径一致：把 80 张图的 dx、dy 合并，RMSE = sqrt((mean(dx²) + mean(dy²))/2)；"
      "交叉验证中每张图恰由未见过它的模型预测一次，得到完整 OOF。在线以 Kaggle 公开榜为准，"
      "每天 5 次提交额度。")

    # ---------------- 3.4 ----------------
    H2("3.4  结果与分析")
    H3("3.4.1  版本演进")
    P("表 3-3 汇总六个版本。最大收益来自 v2 的训练稳定性修复（46 → 18），其次是 v3 的多 seed 集成"
      "（18 → 16）；v4（更强增强）与 v5（CLAHE）在验证集上均未超过 v3，v6 则出现了验证集变差、"
      "榜单反而更好的情况，下文分别讨论。")
    add_table(doc, ["版本", "关键改动", "OOF / px", "榜单 / px"], VER_ROWS,
              widths=[1.6, 8.6, 3.0, 2.6], font_size=8.8,
              caption="表 3-3  版本演进与消融")
    P("v5 的动机是“难样本多为雾状低对比图像”，做法是改用绿通道 CLAHE。视觉上黄斑暗斑确实清晰了许多，"
      "8 张难图平均误差也从 69.7 降到 61.9 像素；但一批原本很准的图像明显变差（0041 由 18.0 升至 37.7，"
      "0023 由 4.4 升至 24.3），整体反而由 16.9 升到 18.8。灰度化丢失颜色信息、CLAHE 放大伪结构，"
      "是净效果为负的两个原因，故最终未采用。")

    H3("3.4.2  训练过程")
    P("图 3-3(a) 是修复前的曲线：崩溃折的验证误差从 176 单调升到 302 像素后被早停；"
      "作为对照，同一次运行中 fold 0 能一路降到 19 像素，说明崩溃来自优化过程而非数据。"
      "图 3-3(b) 是修复后的 5 折：各折在 20~50 轮内收敛到 10~25 像素，未再出现单调上升。")
    FIG(f_curve, "图 3-3  修复前后的验证集收敛曲线", 14.0)

    H3("3.4.3  误差归因")
    P("整体 RMSE 掩盖了误差的真实结构。把最终模型（v6）的 80 个 OOF 误差降序排列（图 3-4），"
      "前 8 张的平方误差占 87.1%，其余 72 张平均仅 10.1 像素、中位数 7.8 像素——模型的典型水平是"
      " 8 像素级，总分被极少数样本拖到 23 像素。这 8 张的共同点不在尺寸（5 张方块图误差均 7~19 像素），"
      "而在成像质量与个体差异：误差与 Laplacian 清晰度的 Spearman 相关系数为 −0.269（p = 0.016）。")
    FIG(f_err, "图 3-4  最终模型的逐图 OOF 误差（红柱为最差的 8 张）", 13.4)
    P("其中 0049 值得单独说明：它在 512 与 640、不同损失与增强的多次实验中，都稳定偏向同一侧"
      " 121~136 像素，方向一致，更像标注本身存在歧义而非模型随机波动；0005 则属于“锁定到错误暗区”，"
      "偏差 162 像素。从图像上看，前者是雾状低对比（清晰度 23.7），后者亮度高达 0.51 且伴有强反光伪影。"
      "这两类错误都无法靠继续调参解决，构成了本任务的精度上限；相比之下，5 张 1956×1934 的方块图"
      "误差均在 7~19 像素之间，说明尺寸本身并不是难点。")

    H3("3.4.4  分辨率与集成的取舍")
    P("同为 5 折 × 3 seed、同样的损失与增强，只把输入从 512 提到 640：验证集由 16.9 恶化到 23.0，"
      "榜单却由 16 改善到 14。验证集的恶化几乎全部来自三张图（0005 由 23 跳到 167、0034 由 81 到 125、"
      "0024 由 23 到 70）；剔除这三张后，剩余 77 张的 RMSE 只有 13.1 像素，反而优于 512。"
      "也就是说 640 对绝大多数图像更好，只是在极少数样本上出现失稳，而测试集恰好不含这类样本。"
      "这提示：在 80 张、误差重尾的条件下，OOF 的方差远大于模型间真实差异，"
      "用它做选择会偏向“在难样本上更保守”的模型。")
    P("集成方面，三个种子的 OOF 分别为 16.61 / 18.41 / 18.76，等权融合为 16.90——优于两个较弱种子，"
      "却不如最好的种子 42。原因是误差相关：三个种子在同一张图上错得一样，平均无法抵消。"
      "因此多 seed 的真实价值在于抑制单次训练崩溃，而不是压低平均误差。")

    # ---------------- 3.5 ----------------
    H2("3.5  创新点与不足")
    P("本实验的贡献不在网络结构，而在“让简单框架在小样本、重尾误差下稳定工作”：")
    NUM(["提出峰值窗口质心解码：先取峰值再在 7×7 窗口内求质心，从机制上消除全图 soft-argmax 的"
         "中心偏置，同时保留亚像素精度，并以合成高斯验证。",
         "把训练崩溃当作可诊断、可拦截的工程问题：偏置初始化 + BCE + warmup 解决热力图饱和，"
         "配合失败折自动重训，使 5 折全部收敛，榜单因此从 46 分降到 18 分。",
         "用逐图误差归因代替整体分数指导迭代：结合清晰度指标与逐图明细，量化出“8 张图贡献 87% 平方误差”，"
         "据此判断继续调参的边际收益很低，避免了无效投入。",
         "发现并解释了验证集与榜单的背离：640 输入在 OOF 上更差、在剔除离群样本后更好、榜单也更好，"
         "说明小样本场景应以榜单为准。"])
    P("不足有三。其一，(0,0) 情形无法学习，模型没有可见性判别能力，只能一律输出坐标。"
      "其二，标注精度构成硬上限：框中心抖动估计 8~12 像素，0049 这类样本疑似标注歧义，继续优化收益有限。"
      "其三，未实现两阶段细化；本文误差主要来自“看错地方”而非“看得不准”，两阶段对粗定位失败无能为力，"
      "但若能把失败率降下来仍有价值。")

    # ---------------- 3.6 ----------------
    H2("3.6  小结")
    P("本章构建了“letterbox 坐标归一化 + ResNet18 热力图回归 + 峰值窗口质心解码 + 5 折 3 seed 集成”的"
      "黄斑中央凹定位方案。在 80 张训练图的条件下，误差由常数基线的 146.6 像素降至公开榜 14 像素。"
      "实验表明：决定成败的是热力图训练稳定性与坐标解码方式，而非更深的网络；"
      "在重尾误差的小样本任务中，验证集分数不足以支撑模型选择，必须结合逐图归因与真实榜单。")

    # ---------------- 参考文献 ----------------
    H2("参考文献")
    refs = [
        "[1] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. CVPR, 2016.",
        "[2] Law H, Deng J. CornerNet: Detecting Objects as Paired Keypoints[C]. ECCV, 2018.",
        "[3] Newell A, Yang K, Deng J. Stacked Hourglass Networks for Human Pose Estimation[C]. ECCV, 2016.",
        "[4] Loshchilov I, Hutter F. Decoupled Weight Decay Regularization[C]. ICLR, 2019.",
        "[5] Zuiderveld K. Contrast Limited Adaptive Histogram Equalization[M]. Graphics Gems IV, 1994.",
    ]
    for r_ in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.9)
        p.paragraph_format.first_line_indent = Cm(-0.9)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        run = p.add_run(r_)
        set_run_font(run, size=8.8)

    doc.save(OUT)
    print("[docx] saved:", OUT)


if __name__ == "__main__":
    build()
