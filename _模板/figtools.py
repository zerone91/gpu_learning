#!/usr/bin/env python3
"""配图工具。规范见 _模板/配图规范.md。在仓库根目录运行。

  python3 _模板/figtools.py search "NAND flash cell" [-n 20]
      在 Wikimedia Commons 检索图片，列出标题、许可、作者、尺寸、来源页。
  python3 _模板/figtools.py get "File:NAND_levels.png" <模块目录>/figures/20-04-vth-levels.png [--width 1400]
      下载 Commons 图片（位图按宽度取缩略图），透明背景填白、SVG 加白底，
      并打印一行可直接粘进图注的“来源：”。
  python3 _模板/figtools.py render <svg 或 png> [--out 目录]
      用 Chrome 无头模式把 SVG 按 viewBox 尺寸渲染成 PNG（默认输出到 /tmp），供亲眼检查。
  python3 _模板/figtools.py check [模块目录 ...]
      检查：图片链接是否存在、figures/ 下有无闲置文件、每张图是否有“来源：”、
      章内图号是否重复、SVG 是否有白底、图注与来源是否分成两段。
"""
import argparse, glob, html, io, json, os, re, subprocess, sys, tempfile, urllib.parse, urllib.request

UA = "gpu-learning-notes/1.0 (figure sourcing; contact via repository owner)"
API = "https://commons.wikimedia.org/w/api.php"
CHROME = next((c for c in ("google-chrome", "chromium", "chromium-browser")
               if subprocess.run(["which", c], capture_output=True).returncode == 0), None)


def _get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return data if binary else json.loads(data)


def _strip(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def _info(titles, width=None):
    q = {"action": "query", "format": "json", "prop": "imageinfo",
         "iiprop": "url|size|mime|extmetadata",
         "iiextmetadatafilter": "LicenseShortName|Artist|ImageDescription",
         "titles": "|".join(titles)}
    if width:
        q["iiurlwidth"] = width
    return _get(API + "?" + urllib.parse.urlencode(q))


def cmd_search(a):
    q = {"action": "query", "format": "json", "generator": "search",
         "gsrnamespace": 6, "gsrlimit": a.n, "gsrsearch": a.query,
         "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
         "iiextmetadatafilter": "LicenseShortName|Artist|ImageDescription"}
    d = _get(API + "?" + urllib.parse.urlencode(q))
    pages = sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    if not pages:
        print("（无结果，换英文关键词或同义词再试）")
    for p in pages:
        ii = p["imageinfo"][0]; m = ii.get("extmetadata", {})
        print(f"{p['title']}\n  许可 {_strip(m.get('LicenseShortName', {}).get('value'))}"
              f" | 作者 {_strip(m.get('Artist', {}).get('value'))[:60]}"
              f" | {ii.get('width')}x{ii.get('height')} {ii.get('mime')}\n"
              f"  描述 {_strip(m.get('ImageDescription', {}).get('value'))[:120]}\n"
              f"  来源页 {ii['descriptionurl']}")


def cmd_get(a):
    title = a.title if a.title.startswith("File:") else "File:" + a.title
    d = _info([title], a.width)
    p = next(iter(d["query"]["pages"].values()))
    if "imageinfo" not in p:
        sys.exit(f"找不到 {title}")
    ii = p["imageinfo"][0]; m = ii.get("extmetadata", {})
    lic = _strip(m.get("LicenseShortName", {}).get("value"))
    artist = _strip(m.get("Artist", {}).get("value"))
    if not re.search(r"public domain|cc0|cc by", lic, re.I):
        print(f"警告：许可为 {lic!r}，不是本库接受的自由许可，请勿入库。")
    is_svg = ii.get("mime") == "image/svg+xml"
    want_svg = a.dest.lower().endswith(".svg")
    url = ii["url"] if (is_svg and want_svg) or not a.width else ii.get("thumburl", ii["url"])
    data = _get(url, binary=True)
    note = ""
    if want_svg:
        s = data.decode("utf-8")
        if not re.search(r'<rect[^>]*fill="(#fff|#ffffff|white)"', s[:4000], re.I):
            s = re.sub(r"(<svg\b[^>]*>)", r'\1<rect x="-10%" y="-10%" width="120%" height="120%" fill="#ffffff"/>', s, count=1)
            note = "（本库加了白色背景）"
        open(a.dest, "w", encoding="utf-8").write(s)
    else:
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            if im.getextrema()[3][0] < 255:
                bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3]); im = bg
                note = "（本库把透明背景填成了白色）"
        im = im.convert("RGB")
        kw = {"quality": 88} if a.dest.lower().endswith((".jpg", ".jpeg")) else {"optimize": True}
        im.save(a.dest, **kw)
    kb = os.path.getsize(a.dest) // 1024
    print(f"已保存 {a.dest}（{kb} KB）")
    if kb > 500:
        print("提示：超过 500 KB，考虑减小 --width 或改存 JPG。")
    name = title[5:].replace("_", " ")
    print(f"> 来源：[{name}]({ii['descriptionurl']})，作者 {artist}，许可 {lic}{note}。")


def cmd_render(a):
    if not CHROME:
        sys.exit("没有找到 Chrome/Chromium")
    src = os.path.abspath(a.file)
    w, h = 1000, 700
    if src.endswith(".svg"):
        head = open(src, encoding="utf-8").read(3000)
        vb = re.search(r'viewBox="\s*[-\d.]+[ ,]+[-\d.]+[ ,]+([\d.]+)[ ,]+([\d.]+)', head)
        wh = re.search(r'<svg\b[^>]*?\bwidth="([\d.]+)(?:px)?"[^>]*?\bheight="([\d.]+)(?:px)?"', head, re.S)
        if vb:
            w, h = int(float(vb.group(1))), int(float(vb.group(2)))
        elif wh:
            w, h = int(float(wh.group(1))), int(float(wh.group(2)))
    out = os.path.join(a.out or tempfile.gettempdir(), os.path.splitext(os.path.basename(src))[0] + ".render.png")
    # 无头 Chrome 的可视区域比 --window-size 矮约 90 像素，直接用原尺寸会裁掉图的底部。
    # 所以窗口加高 300 像素渲染，再裁回原尺寸。
    subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                    f"--screenshot={out}", f"--window-size={w},{h + 300}", "file://" + src],
                   capture_output=True)
    if not os.path.exists(out):
        sys.exit("渲染失败")
    from PIL import Image
    Image.open(out).crop((0, 0, w, h)).save(out)
    print(out)


def cmd_check(a):
    dirs = a.dirs or sorted(d for d in glob.glob("[0-9]-*/[0-9][0-9]-*") if os.path.isdir(d))
    problems = 0
    for d in dirs:
        mds = sorted(glob.glob(os.path.join(d, "*.md")))
        figdir = os.path.join(d, "figures")
        used = set()
        for f in mds:
            lines = open(f, encoding="utf-8").read().split("\n")
            nums = []
            for i, l in enumerate(lines):
                for p in re.findall(r"!\[(?:[^\[\]]|\[[^\[\]]*\])*\]\((\./figures/[^)\s]+)\)", l):
                    used.add(os.path.basename(p))
                    if not os.path.exists(os.path.join(d, p)):
                        print(f"[缺文件] {f}:{i+1} {p}"); problems += 1
                    blk = lines[i + 1:i + 60]
                    if not any(x.startswith("> 来源：") for x in blk):
                        print(f"[无来源] {f}:{i+1}"); problems += 1
                m = re.match(r"> \*\*图 ([A-Z]?\d+-\d+)\*\*", l)
                if m:
                    nums.append(m.group(1))
                if l.startswith("> 来源：") and i and lines[i - 1].startswith("> ") and lines[i - 1].strip() != ">":
                    print(f"[图注与来源未分段] {f}:{i+1}（在两行之间加一行 '>'）"); problems += 1
            dup = {n for n in nums if nums.count(n) > 1}
            if dup:
                print(f"[图号重复] {f} {sorted(dup)}"); problems += 1
        if os.path.isdir(figdir):
            for p in sorted(os.listdir(figdir)):
                fp = os.path.join(figdir, p)
                if p not in used:
                    print(f"[闲置文件] {fp}"); problems += 1
                if p.endswith(".svg") and not re.search(r'fill="(#fff|#ffffff|white)"|background',
                                                        open(fp, encoding="utf-8").read(4000), re.I):
                    print(f"[SVG 无白底] {fp}"); problems += 1
                if os.path.getsize(fp) > 500 * 1024:
                    print(f"[文件偏大] {fp} {os.path.getsize(fp)//1024} KB")
            print(f"{d}: {len(used)} 张图被引用，figures/ 共 {len(os.listdir(figdir))} 个文件")
    print("检查通过" if problems == 0 else f"共 {problems} 个问题")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("search"); s.add_argument("query"); s.add_argument("-n", type=int, default=20); s.set_defaults(f=cmd_search)
    g = sp.add_parser("get"); g.add_argument("title"); g.add_argument("dest"); g.add_argument("--width", type=int, default=1400); g.set_defaults(f=cmd_get)
    r = sp.add_parser("render"); r.add_argument("file"); r.add_argument("--out"); r.set_defaults(f=cmd_render)
    c = sp.add_parser("check"); c.add_argument("dirs", nargs="*"); c.set_defaults(f=cmd_check)
    a = ap.parse_args()
    sys.exit(1 if a.f(a) else 0)
