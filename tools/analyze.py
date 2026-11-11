"""误差归因分析：读取 oof_detail.csv，输出报告 3.4.3 节用到的统计量。

用法：
    python tools/analyze.py results/v6_640_3seed/oof_detail.csv

若仓库根目录下存在 raw/detection/，会额外计算每张图的清晰度与亮度。
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "raw", "detection")


def quality(path, size=512):
    import cv2
    im = cv2.imread(path)
    if im is None:
        return 0.0, 0.0
    h, w = im.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(im, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                    interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = g > 12
    return float((g[m] / 255).mean()), float(cv2.Laplacian(g, cv2.CV_32F)[m].var())


def main(csv_path):
    d = pd.read_csv(csv_path, dtype={"id": str})
    sse = (d.dist ** 2).values
    oof = np.sqrt(sse.sum() / (2 * len(d)))
    top = np.sort(sse)[::-1]

    print("样本数        : %d" % len(d))
    print("OOF RMSE      : %.2f px" % oof)
    print("前 8 张 SSE 占比: %.1f%%" % (100 * top[:8].sum() / sse.sum()))
    print("其余 %d 张     : 均值 %.1f px，中位数 %.1f px"
          % (len(d) - 8, d.dist.sort_values().iloc[:-8].mean() if len(d) > 8 else 0,
             d.dist.sort_values().iloc[:-8].median() if len(d) > 8 else 0))
    print("若前 8 张均为 20 px，OOF 将降为 %.2f px"
          % np.sqrt((top[8:].sum() + 8 * 400) / (2 * len(d))))

    print("\n误差最大的 10 张：")
    cols = ["id", "dx", "dy", "dist"]
    print(d.sort_values("dist", ascending=False).head(10)[cols].round(1).to_string(index=False))

    if os.path.isdir(DATA):
        rows = []
        for _, r in d.iterrows():
            b, sh = quality(os.path.join(DATA, "train", r.id + ".jpg"))
            rows.append((r.dist, b, sh))
        arr = np.array(rows)
        from scipy.stats import spearmanr
        rho, p = spearmanr(arr[:, 0], arr[:, 2])
        print("\n误差与清晰度 Spearman rho = %.3f (p = %.4f)" % (rho, p))
        rho2, p2 = spearmanr(arr[:, 0], arr[:, 1])
        print("误差与亮度   Spearman rho = %.3f (p = %.4f)" % (rho2, p2))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "results", "v6_640_3seed", "oof_detail.csv")
    main(target)
