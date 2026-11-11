"""把报告里的代码仓库链接替换成实际地址（不需要数据即可运行）。

用法：
    python report/set_github_url.py https://github.com/<你的用户名>/<仓库名>

会同时修改 report/build_report.py 中的常量，以及已生成的 docx 中出现的旧链接。
"""
import os
import sys

from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = "https://github.com/DaoTongx/fovea-localization"
DOCX = os.path.join(HERE, "任务三_眼底黄斑定位_实验报告.docx")
BUILD = os.path.join(HERE, "build_report.py")


def main(url):
    url = url.rstrip("/")
    if os.path.exists(BUILD):
        txt = open(BUILD, encoding="utf-8").read()
        open(BUILD, "w", encoding="utf-8").write(txt.replace(OLD, url))
        print("[build] 已更新", BUILD)

    if os.path.exists(DOCX):
        doc = Document(DOCX)
        n = 0
        for p in doc.paragraphs:
            for r in p.runs:
                if OLD in r.text:
                    r.text = r.text.replace(OLD, url)
                    n += 1
        doc.save(DOCX)
        print("[docx] 已替换 %d 处链接" % n)
    print("完成：", url)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
