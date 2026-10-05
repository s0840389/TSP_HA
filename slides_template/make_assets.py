"""Build the BoE logo variants from boe/BoE_logo.png and embed them as base64 CSS variables,
so the HTML slides are a single self-contained file.

Run after changing the logo:  python slides_template/make_assets.py
"""
import base64
from pathlib import Path

from PIL import Image, ImageDraw

here = Path(__file__).parent / "boe"
NAVY = (18, 39, 63)


def make_variants():
    im = Image.open(here / "BoE_logo.png").convert("RGBA")
    w, h = im.size

    # transparent background: flood-fill the white surround from the corners
    mask = im.convert("RGB")
    for xy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        ImageDraw.floodfill(mask, xy, (255, 0, 255), thresh=60)
    px, m = im.load(), mask.load()
    for y in range(h):
        for x in range(w):
            if m[x, y] == (255, 0, 255):
                px[x, y] = (255, 255, 255, 0)
    lockup = im.crop(im.getbbox())
    lockup.save(here / "boe_logo.png", optimize=True)

    # roundel only: everything above the first empty row (the gap before the wordmark)
    alpha = lockup.split()[3]
    lw, lh = lockup.size
    gap = next(y for y in range(lh) if all(alpha.getpixel((x, y)) == 0 for x in range(lw)))
    roundel = lockup.crop((0, 0, lw, gap))
    roundel.crop(roundel.getbbox()).save(here / "boe_roundel.png", optimize=True)

    # white lockup for the navy title slide: navy -> white, white -> navy
    white = lockup.copy()
    q = white.load()
    for y in range(lh):
        for x in range(lw):
            r, g, b, a = q[x, y]
            if a == 0:
                continue
            t = ((r + g + b) / 3 - NAVY[0]) / (255 - NAVY[0])
            q[x, y] = tuple(int(255 + (c - 255) * t) for c in NAVY) + (a,)
    white.save(here / "boe_logo_white.png", optimize=True)


def embed():
    assets = {
        "--boe-roundel": "boe_roundel.png",          # navy roundel (watermark + header bar)
        "--boe-logo-white": "boe_logo_white.png",    # white lockup for the navy title slide
    }
    lines = [":root {"]
    for var, name in assets.items():
        b64 = base64.b64encode((here / name).read_bytes()).decode()
        lines.append(f'  {var}: url("data:image/png;base64,{b64}");')
    lines.append("}")
    (here / "static" / "boe_assets.css").write_text("\n".join(lines) + "\n")
    print("wrote", here / "static" / "boe_assets.css")


if __name__ == "__main__":
    make_variants()
    embed()
