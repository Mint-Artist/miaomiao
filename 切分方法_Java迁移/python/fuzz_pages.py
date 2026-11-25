"""随机测试页生成器：覆盖 5 种方法的全部规则和 Python / Java 容易不一致的地方（固定种子，可复现）。

    from fuzz_pages import gen_html, gen_markdown
    html = gen_html(random.Random(0))
"""
import random

ZH = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经"
TRICKY = ["😀", "𠀀", "１", "２０２５", " ", "　", "\t", " ", "Ｓｔｅｐ", "ſtep", "&nbsp;", "&amp;", "&lt;", "Ⅻ"]
NUM_PREFIX = ["1、", "2.", "3．", "4)", "5）", "(6)", "（7）", "一、", "二.", "（三）", "(四)", "第一章 ", "第3步 ", "第二部分 ",
              "①", "⑤ ", "Step 1 ", "步骤 2 ", "1.5亿", "10.5%", "１、", "2023年"]
ENDS = ["。", "！", "？", "；", "，", "、", "…", "：", ":", "", "", "", "!", "?"]


def words(rng, a, b):
    n = rng.randint(a, b)
    s = "".join(rng.choice(ZH) for _ in range(n))
    if rng.random() < 0.15:
        s = s[: len(s) // 2] + rng.choice(TRICKY) + s[len(s) // 2:]
    return s


def sentence(rng, a=5, b=40):
    return words(rng, a, b) + rng.choice(["。", "。", "，", "！", "？", "；", "…", "。”", "）。"])


def para_text(rng):
    r = rng.random()
    if r < 0.05:
        return words(rng, 1100, 1500)                                     # 超长、没有标点
    if r < 0.12:
        return "".join(sentence(rng, 20, 80) for _ in range(rng.randint(8, 25)))  # 超长、有标点
    return "".join(sentence(rng) for _ in range(rng.randint(1, 8)))


def title_line(rng):
    return rng.choice(NUM_PREFIX) + words(rng, 2, 18) + rng.choice(ENDS)


def inline(rng, text):
    r = rng.random()
    if r < 0.1:
        return f'<a href="https://x.example/{rng.randint(1, 99)}">{text}</a>'
    if r < 0.18:
        return f"<span>{text}</span><em>{words(rng, 1, 4)}</em>"
    if r < 0.22:
        return f'{text}<img src="i.png">{words(rng, 1, 5)}'
    if r < 0.26:
        return text + "<br>" + words(rng, 2, 20)
    if r < 0.28:
        return text + "<!-- 注释" + words(rng, 1, 5) + " -->"
    return text


# ---------------- html ----------------
def h_list(rng, depth=0):
    tag = rng.choice(["ul", "ol"])
    items = []
    for _ in range(rng.randint(1, 12)):
        it = inline(rng, rng.choice([words(rng, 2, 30), sentence(rng, 5, 120), title_line(rng)]))
        if rng.random() < 0.2:
            it += f"<p>{sentence(rng)}</p>"
        if depth < 2 and rng.random() < 0.12:
            it += h_list(rng, depth + 1)
        items.append(f"<li>{it}</li>" if rng.random() > 0.05 else f"<li>{it}")  # 偶尔不闭合
    return f"<{tag}>" + "".join(items) + f"</{tag}>"


def h_table(rng, depth=0):
    r = rng.random()
    if r < 0.2:  # 排版用单列表格
        return "<table>" + "".join(f"<tr><td>{inline(rng, para_text(rng))}</td></tr>" for _ in range(rng.randint(1, 3))) + "</table>"
    cols = rng.randint(2, 5)
    head = ""
    if rng.random() < 0.5:
        th = "".join(f"<th>{words(rng, 1, 6)}</th>" for _ in range(cols))
        head = f"<thead><tr>{th}</tr></thead>" if rng.random() < 0.5 else f"<tr>{th}</tr>"
    cap = f"<caption>{words(rng, 2, 10)}</caption>" if rng.random() < 0.3 else ""
    rows = []
    for _ in range(rng.randint(1, 45)):
        cells = []
        for _ in range(cols):
            c = rng.choice([words(rng, 1, 25), "", str(rng.randint(1, 9999)), sentence(rng, 3, 30)])
            if depth == 0 and rng.random() < 0.01:
                c += h_table(rng, 1)
            cells.append(f"<td>{c}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(rows)
    if rng.random() < 0.3:
        body = f"<tbody>{body}</tbody>"
    return f"<table>{cap}{head}{body}</table>"


def h_block(rng, depth=0):
    r = rng.random()
    if r < 0.08:
        lv = rng.randint(1, 6)
        t = words(rng, 120, 160) if rng.random() < 0.05 else words(rng, 2, 20)
        return f"<h{lv}>{inline(rng, t)}</h{lv}>"
    if r < 0.13:
        return f"<p><strong>{words(rng, 2, 20)}{rng.choice(ENDS)}</strong></p>"
    if r < 0.22:
        return f"<p>{title_line(rng)}</p>"
    if r < 0.32:
        return h_list(rng)
    if r < 0.40:
        return h_table(rng)
    if r < 0.43:
        return f'<div class="note">{rng.choice([h_list(rng), "<p>" + sentence(rng) + "</p>", words(rng, 2, 8)])}</div>'
    if r < 0.45:
        return '<div class="metadata">' + "".join(f"<p>{words(rng, 2, 6)}：{words(rng, 2, 40)}</p>" for _ in range(2)) + "</div>"
    if r < 0.47:
        return "<pre>" + "\n".join("x = " + words(rng, 1, 30) for _ in range(rng.randint(1, 60))) + "</pre>"
    if r < 0.49:
        return f"<blockquote><p>{sentence(rng)}</p><p>{sentence(rng)}</p></blockquote>"
    if r < 0.50:
        return f'<figure><img src="a.png"><figcaption>{words(rng, 3, 20)}</figcaption></figure>'
    if r < 0.54 and depth < 3:
        tag = rng.choice(["div", "section", "article", "div"])
        return f"<{tag}>" + "".join(h_block(rng, depth + 1) for _ in range(rng.randint(1, 4))) + f"</{tag}>"
    if r < 0.56:
        return inline(rng, words(rng, 3, 40))                              # 外壳里直接出现的文字
    if r < 0.57:
        return "<script>var a = '不应出现';</script><style>.x{}</style>"
    if r < 0.58:
        return f"<p>{sentence(rng)}"                                       # 不闭合的 p
    if r < 0.59:
        return f"<div><p>{sentence(rng)}</div>"                            # 错配
    if r < 0.60:
        return f"<P>{sentence(rng)}</P>"                                   # 大写标签
    return f"<p>{inline(rng, para_text(rng))}</p>"


def gen_html(rng):
    pre = "<!DOCTYPE html>" if rng.random() < 0.1 else ""
    lead = words(rng, 2, 10) if rng.random() < 0.05 else ""                # 开头裸文本
    body = "\n".join(h_block(rng) for _ in range(rng.randint(1, 40)))
    if rng.random() < 0.3:
        body = body.replace("\n", "\r\n")
    return pre + lead + body


# ---------------- markdown ----------------
def m_inline(rng, t):
    r = rng.random()
    if r < 0.08:
        return f"[{t}](https://x.example/{rng.randint(1, 9)})"
    if r < 0.12:
        return f"![{words(rng, 1, 4)}](a.png){t}"
    if r < 0.18:
        return f"**{t}**" + words(rng, 0, 5)
    if r < 0.21:
        return t + "<br>" + words(rng, 2, 10)
    if r < 0.23:
        return f"<span>{t}</span>"
    return t


def m_block(rng):
    r = rng.random()
    if r < 0.1:
        return "#" * rng.randint(1, 6) + " " + m_inline(rng, words(rng, 2, 20)) + (" ##" if rng.random() < 0.1 else "")
    if r < 0.15:
        return f"**{words(rng, 2, 20)}{rng.choice(ENDS)}**"
    if r < 0.24:
        return title_line(rng)
    if r < 0.36:
        mark = rng.choice(["-", "*", "+", "1.", "2)"])
        return "\n".join(f"{'  ' if rng.random() < 0.2 else ''}{mark} {m_inline(rng, rng.choice([words(rng, 2, 30), sentence(rng, 5, 120), title_line(rng)]))}"
                         for _ in range(rng.randint(1, 10)))
    if r < 0.44:
        cols = rng.randint(2, 4)
        head = "| " + " | ".join(words(rng, 1, 5) for _ in range(cols)) + " |"
        sep = "|" + "|".join(rng.choice(["---", ":--", "--:", ":-:"]) for _ in range(cols)) + "|"
        rows = ["| " + " | ".join(m_inline(rng, words(rng, 1, 20)) for _ in range(cols)) + " |" for _ in range(rng.randint(1, 40))]
        return "\n".join([head, sep] + rows)
    if r < 0.50:
        return "<table>\n" + h_table(rng).replace("<tr>", "\n<tr>") + "\n</table>" if rng.random() < 0.3 else h_table(rng)
    if r < 0.54:
        return "```\n" + "\n".join("y = " + words(rng, 1, 30) for _ in range(rng.randint(1, 50))) + "\n```"
    if r < 0.56:
        return words(rng, 2, 10) + "：" if rng.random() < 0.5 else "> " + sentence(rng)
    return m_inline(rng, para_text(rng))


def gen_markdown(rng):
    md = "\n\n".join(m_block(rng) for _ in range(rng.randint(1, 40)))
    if rng.random() < 0.3:
        md = md.replace("\n", "\r\n")
    return md
