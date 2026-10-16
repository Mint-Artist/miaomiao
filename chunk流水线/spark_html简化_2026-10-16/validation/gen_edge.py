"""生成边界情况测试输入 edge.jsonl（其中 huge 一例约 18MB，所以只保留脚本）。
用法：python3 gen_edge.py && scala-cli run ../HtmlSimplify.scala -- --local --input edge.jsonl --output edge_out.jsonl
"""
import json

CASES = [
 {
  "id": "empty",
  "url": "http://a.com/1",
  "pg": ""
 },
 {
  "id": "nofield",
  "url": "http://a.com/2"
 },
 {
  "id": "plaintext",
  "url": "http://a.com/3",
  "pg": "这不是HTML，只是一段纯文本。第二句话。"
 },
 {
  "id": "broken",
  "url": "http://a.com/6",
  "pg": "<html><body><div><p>未闭合的段落<p>第二段，内容足够长，用来测试容错解析。<table><tr><td>格子<div>块</td></body>"
 },
 {
  "id": "double",
  "url": "http://a.com/7",
  "pg": "\"\\\"<html><body><article><h1>标题</h1><p>两层 JSON 编码的正文，内容应该被正确解开。</p></article></body></html>\\\"\""
 },
 {
  "id": "images",
  "url": "http://a.com/8",
  "pg": "<html><body><div class=\"content\"><img data-original=\"/a.jpg\" src=\"grey.gif\" alt=\"图一\"><img src=\"data:image/png;base64,xx\"></div></body></html>"
 },
 {
  "id": "brparas",
  "url": "http://a.com/9",
  "pg": "<html><body><div id=\"content\">　　第一段正文，讲了一些事情。<br>　　第二段正文，继续讲述，内容比较长一些。<br><br>第三段，前面有两个换行。<br>同一段内的换行。</div></body></html>"
 },
 {
  "id": "nested",
  "url": "http://a.com/10",
  "pg": "<html><body><div class=\"article\"><p>正文开头，有一些说明文字。</p><table><tr><th>项目</th><th>数值</th></tr><tr><td rowspan=\"2\">甲</td><td>1</td></tr><tr><td>2</td></tr></table><ul><li>第一项，带有说明。<ul><li>子项一</li><li>子项二</li></ul></li><li>第二项，也有说明。</li></ul><table><tr><td><table><tr><td>嵌套表格里的正文段落，较长的一段文字，应当按布局表格处理。</td></tr></table></td></tr></table></div></body></html>"
 },
 {
  "id": "hidden",
  "url": "http://a.com/11",
  "pg": "<html><body><div class=\"content\"><p>可见的正文段落，包含足够多的文字和标点。</p><div style=\"display:none\">隐藏的广告文字</div><p hidden>隐藏段落</p></div></body></html>"
 },
 {
  "id": "aspx",
  "url": "http://gov.cn/x.aspx",
  "pg": "<html><body><form id=\"form1\"><div class=\"nav\"><a href=\"/\">首页</a> <a href=\"/a\">新闻</a></div><div class=\"TRS_Editor\"><p>政府网站的正文第一段，内容是通知公告。</p><p>正文第二段，说明了具体的办理流程和要求。</p></div></form></body></html>"
 },
 {
  "id": "frameset",
  "url": "http://a.com/12",
  "pg": "<html><frameset><frame src=\"a.html\"></frameset></html>"
 },
 {
  "id": "script",
  "url": "http://a.com/13",
  "pg": "<html><body><script>var a=\"很多很多文字。\";</script><noscript>请开启JavaScript</noscript></body></html>"
 },
 {
  "id": "badurl",
  "url": "not a url",
  "pg": "<html><body><p>链接<a href='/rel'>相对</a>和<a href='javascript:void(0)'>脚本</a>，正文内容。</p></body></html>"
 },
 {
  "id": "pre",
  "url": "http://a.com/14",
  "pg": "<html><body><article><p>下面是代码，内容很重要，需要保留原样。</p><pre>def f(x):\n    return x  # 缩进保留</pre></article></body></html>"
 }
]
CASES.append({'id': 'huge', 'url': 'http://a.com/4', 'pg': '<p>' + '很长的文字。' * 1000000 + '</p>'})
CASES.append({'id': 'deep', 'url': 'http://a.com/5', 'pg': '<html><body>' + '<div>' * 20000 +
              '<p>深层正文内容，用来测试极深嵌套不会导致栈溢出。</p>' + '</div>' * 20000 + '</body></html>'})

with open('edge.jsonl', 'w', encoding='utf-8') as f:
    for c in CASES:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
print(len(CASES))
