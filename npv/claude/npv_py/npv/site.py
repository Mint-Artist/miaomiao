"""站点解析。

Java 侧使用的是原项目的 ParseSiteUtil.parseSite，PDF 中没有其实现。
这里给出一个合理的默认实现：取 URL 的 host，小写，去掉端口，不去掉 www 前缀。
如果原实现有差异（例如会去掉 www、或保留大小写），请在这里对齐，
它同时影响 dr / ow / 黑白名单四处匹配。
"""
from urllib.parse import urlparse


def parse_site(url: str) -> str:
    u = url.strip()
    if "://" not in u:
        u = "http://" + u
    try:
        host = urlparse(u).hostname
    except ValueError:
        host = None
    return (host or "").lower()
