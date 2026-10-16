"""validation/compare_viewer.html：逐页并排查看现有链路（Trafilatura + 自研简化）与 HtmlSimplify.scala 的输出。

用法：python3 make_viewer.py out_final.jsonl
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / '开源对照_2026-09-28' / 'results'


def main():
    out = sys.argv[1]
    per = json.loads((HERE / (Path(out).stem + '_compare.json')).read_text()) if (HERE / (Path(out).stem + '_compare.json')).exists() else {}
    pages = []
    for line in (HERE / out).open(encoding='utf-8'):
        d = json.loads(line)
        ref_path = SRC / 'pages' / d['id'] / 'simplified.html'
        ref = re.sub(r' data-src="[^"]*"', '', ref_path.read_text(encoding='utf-8')) if ref_path.exists() else ''
        p = per.get(d['id'], {})
        pages.append({'id': d['id'], 'url': d['url'], 'status': d['status'], 'locate': d['locate'],
                      'inferred': d['inferredHeadings'], 'chars': d['textChars'],
                      'recall': p.get('recall'), 'precision': p.get('precision'), 'ref': ref, 'new': d['html']})
    data = json.dumps(pages, ensure_ascii=False).replace('</', '<\\/')
    html = PAGE.replace('__DATA__', data)
    (HERE / 'compare_viewer.html').write_text(html, encoding='utf-8')
    print(HERE / 'compare_viewer.html', len(pages))


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>简版 HTML 对比</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#fcfcfb;--fg:#1b1b1a;--muted:#6b6a66;--line:#e4e3de;--card:#fff;--hi:#fff4d6;--inf:#1baf7a;--bad:#e34948}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--card:#222220;--hi:#3a3420;--inf:#3ccf95;--bad:#e66767}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"PingFang SC","Noto Sans CJK SC",sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line)}h1.t{font-size:17px;margin:0 0 4px}
.note{color:var(--muted);font-size:12.5px}
.wrap{display:grid;grid-template-columns:300px 1fr;height:calc(100vh - 70px)}
#list{overflow:auto;border-right:1px solid var(--line)}
#list input{width:calc(100% - 16px);margin:8px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
.item{padding:6px 10px;border-bottom:1px solid var(--line);cursor:pointer;font-size:12.5px}
.item:hover,.item.on{background:var(--hi)}.item .u{color:var(--muted);word-break:break-all}
.item .m{font-variant-numeric:tabular-nums}.low{color:var(--bad)}
.panes{display:grid;grid-template-columns:1fr 1fr;overflow:hidden}
.pane{overflow:auto;padding:12px 16px;border-right:1px solid var(--line)}
.pane h2.cap{font-size:13px;color:var(--muted);margin:0 0 8px;position:sticky;top:-12px;background:var(--bg);padding:6px 0}
.doc>*{border-left:3px solid var(--line);padding-left:8px;margin:6px 0}
.doc [data-origin=inferred]{border-left-color:var(--inf)}
.doc [data-origin=inferred]::after{content:" 推断";font-size:11px;color:var(--inf);font-weight:normal}
.doc img{max-width:160px;max-height:90px;vertical-align:middle}
.doc table{border-collapse:collapse}.doc td,.doc th{border:1px solid var(--line);padding:2px 6px;font-size:12.5px}
.doc h1{font-size:18px}.doc h2{font-size:16px}.doc h3,.doc h4,.doc h5,.doc h6{font-size:14.5px}
@media (max-width:820px){.wrap{grid-template-columns:1fr;height:auto}#list{max-height:40vh}.panes{grid-template-columns:1fr}}
</style></head><body>
<header><h1 class="t">简版 HTML 对比：现有链路 vs HtmlSimplify.scala</h1>
<div class="note">左栏：页面列表（召回 / 精确 = 两边正文 8 字片段的互相覆盖，红色 &lt; 0.8）。每个块左侧有竖线；绿色竖线 + "推断" = 由加粗 / 字号 / 编号等推断出的小标题。</div></header>
<div class="wrap"><div id="list"><input id="q" placeholder="按 URL / id 过滤"><div id="items"></div></div>
<div class="panes"><div class="pane"><h2 class="cap">现有链路（Trafilatura + 自研简化）</h2><div class="doc" id="ref"></div></div>
<div class="pane"><h2 class="cap" id="newcap">HtmlSimplify.scala</h2><div class="doc" id="new"></div></div></div></div>
<script>
const P=__DATA__;
const f=x=>x==null?'—':x.toFixed(2);
function render(){const q=document.getElementById('q').value.trim();const box=document.getElementById('items');box.innerHTML='';
P.forEach((p,i)=>{if(q&&!(p.url.includes(q)||p.id.includes(q)))return;const d=document.createElement('div');d.className='item';d.dataset.i=i;
d.innerHTML=`<div class="m"><span class="${p.recall!=null&&p.recall<0.8?'low':''}">召回 ${f(p.recall)}</span> · <span class="${p.precision!=null&&p.precision<0.8?'low':''}">精确 ${f(p.precision)}</span> · ${p.status} · ${p.locate} · 推断 ${p.inferred}</div><div class="u">${p.url}</div>`;
d.onclick=()=>show(i);box.appendChild(d)})}
function show(i){const p=P[i];document.querySelectorAll('.item').forEach(e=>e.classList.toggle('on',+e.dataset.i===i));
document.getElementById('ref').innerHTML=p.ref||'<p style="color:var(--muted)">（现有链路没有产出）</p>';
document.getElementById('new').innerHTML=p.new||'<p style="color:var(--muted)">（空）</p>';
document.getElementById('newcap').textContent=`HtmlSimplify.scala（${p.status}，定位 ${p.locate}，正文 ${p.chars} 字）`}
document.getElementById('q').oninput=render;render();show(0);
</script></body></html>'''

if __name__ == '__main__':
    main()
