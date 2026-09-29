#!/usr/bin/env python3
"""从各节「术语卡」自动生成 0-导览/01-真实芯片与术语表/术语表.md 索引。在仓库根目录运行: python3 _模板/build_index.py"""
import re, glob

files = sorted(glob.glob("[0-9]-*/[0-9][0-9]-*/[0-9AB]*.md"))

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
    mod = f.split('/')[1]
    if mod != cur:
        out.append(f"\n## {mod}\n"); cur = mod
    out.append(f"\n**[{title}](../../{f})**")
    out += [f"- {c}" for c in cards]
total = sum(len(c) for _,c in index.values())
out.append(f"\n---\n\n*共 {total} 张术语卡，覆盖 {len(index)} 节。生成时间见 git 记录。*")
open("0-导览/01-真实芯片与术语表/术语表.md","w").write("\n".join(out)+"\n")
print(f"{total} cards / {len(index)} files")
