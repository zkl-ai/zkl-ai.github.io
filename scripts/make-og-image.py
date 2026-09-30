#!/usr/bin/env python3
"""生成 og 分享图（1200x630）。

改了名字、职位或域名之后重跑一次即可：
    python3 scripts/make-og-image.py
输出：public/og.png
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
PAD = 90

BG = "#ffffff"
TEXT = "#111827"
MUTED = "#5b6472"
FAINT = "#8a93a3"
ACCENT = "#4f46e5"
ACCENT_SOFT = "#eef0ff"
BORDER = "#e6e6ef"

NAME_ZH = "张坤龙"
NAME_EN = "Zhangkunlong"
ROLE = "多模态大模型推理加速 · AI Agent 工程师"
PILL = "AI 工程 · AI Engineering"
DOMAIN = "zkl-ai.top"

CJK_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Light.ttc",
]
LATIN_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def load_font(candidates, size, index=0):
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=index)
            except OSError:
                continue
    raise SystemExit(f"找不到可用字体，试过：{candidates}")


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    cjk = lambda s: load_font(CJK_CANDIDATES, s)
    latin = lambda s: load_font(LATIN_CANDIDATES, s)

    f_pill = cjk(28)
    f_name = cjk(112)
    f_en = latin(42)
    f_role = cjk(36)
    f_domain = latin(28)

    # 左侧品牌色竖条
    d.rectangle([0, 0, 10, H], fill=ACCENT)

    # 顶部药丸标签
    tb = d.textbbox((0, 0), PILL, font=f_pill)
    pw, ph = tb[2] - tb[0], tb[3] - tb[1]
    px, py = PAD, 92
    d.rounded_rectangle(
        [px, py, px + pw + 44, py + ph + 28], radius=24, fill=ACCENT_SOFT
    )
    d.text((px + 22, py + 12 - tb[1]), PILL, font=f_pill, fill=ACCENT)

    # 中文名
    y = 186
    d.text((PAD, y), NAME_ZH, font=f_name, fill=TEXT)
    nb = d.textbbox((PAD, y), NAME_ZH, font=f_name)

    # 英文名跟在中文名右边
    d.text((nb[2] + 28, nb[3] - 52), NAME_EN, font=f_en, fill=FAINT)

    # 职位
    d.text((PAD, 372), ROLE, font=f_role, fill=MUTED)

    # 分隔线
    d.line([PAD, 486, W - PAD, 486], fill=BORDER, width=2)

    # 底部：左域名，右 GitHub
    d.text((PAD, 528), DOMAIN, font=f_domain, fill=FAINT)

    gh = "github.com/zkl-ai"
    gb = d.textbbox((0, 0), gh, font=f_domain)
    d.text((W - PAD - (gb[2] - gb[0]), 528), gh, font=f_domain, fill=FAINT)

    out = Path(__file__).resolve().parent.parent / "public" / "og.png"
    img.save(out, "PNG", optimize=True)
    print(f"✓ 已生成 {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
