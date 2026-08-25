#!/usr/bin/env python3
"""从各节「术语卡」自动生成 05/术语表.md 索引。在仓库根目录运行: python3 _模板/build_index.py"""
import re, glob

files = []
for pat in ["00-*/[0-9]*.md","01-*/[0-9A]*.md","02-*/[0-9]*.md","03-*/[0-9]*.md","04-*/[0-9AB]*.md","05-*/01*.md","06-*/[0-9]*.md","07-*/[0-9]*.md"]:
    files += sorted(glob.glob(pat))

index = {}
for f in files:
    s = open(f).read()
    title = s.splitlines()[0].lstrip('# ').strip()
    m = re.search(r'^## [\d\.]*\s*(?:一句话)?(?:术语卡|黑话卡).*?$', s, re.M)
    cards = []
    if m:
        rest = s[m.end():]
        nxt = re.search(r'^## ', rest, re.M)
        body = rest[:nxt.start()] if nxt else rest
        cards = [c.strip() for c in re.findall(r'^### (.+)$', body, re.M)]
    if cards:
        index[f] = (title, cards)

out = ["# 术语索引\n",
"> 本文件由脚本从各节的「术语卡」自动生成（`_模板/build_index.py`），**不要手工编辑**。每个词条的完整卡片（定义 / 为什么存在 / 语境例句）在对应章节末尾。用编辑器的搜索功能查词即可。\n"]
cur = None
for f,(title,cards) in index.items():
    mod = f.split('/')[0]
    if mod != cur:
        out.append(f"\n## {mod}\n"); cur = mod
    out.append(f"\n**[{title}](../{f})**")
    out += [f"- {c}" for c in cards]
total = sum(len(c) for _,c in index.values())
out.append(f"\n---\n\n*共 {total} 张术语卡，覆盖 {len(index)} 节。生成时间见 git 记录。*")
open("05-收敛-术语表与真实芯片/术语表.md","w").write("\n".join(out)+"\n")
print(f"{total} cards / {len(index)} files")
