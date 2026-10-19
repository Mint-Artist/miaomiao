# 批量提取审计报告

- 输入：235 个网页，127 个 host
- 状态：{'ok': 208, 'failed': 20, 'degraded': 7}
- 提取模式：{'lvtu_zs': 6, 'generic': 208, 'unknown': 20, 'huawei_support': 1}
- 未找到标题：1 页
- 简版/原始字符比中位数：0.0424
- 简版正文字符数中位数：988
- 建议优先抽检（正文过短、压缩比异常或 body 兜底）：18 页

> 状态成功只表示未发生解析异常，不等于正文提取质量。质量结论必须结合人工抽检。

## 失败页

| doc_id | host | error | url |
|---|---|---|---|
| 16bb44826c5e | m.csai.cn | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://m.csai.cn/stock/1458873.html |
| 19b26cc0ff25 | easylearn.baidu.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://easylearn.baidu.com/edu-page/tiangong/questiondetail?id=1741409085705997382 |
| 1dc44fa1e444 | caifuhao.eastmoney.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://caifuhao.eastmoney.com/news/20260622153519300632420 |
| 321ec3aa25c0 | caifuhao.eastmoney.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://caifuhao.eastmoney.com/news/20250222104038091779730 |
| 32748878a5a5 | www.toutiao.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://www.toutiao.com/article/7425884160918618659/ |
| 3d1d80df2f99 | m.toutiao.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://m.toutiao.com/article/7660033310386635314/ |
| 417a1142cde4 | easylearn.baidu.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://easylearn.baidu.com/edu-page/tiangong/questiondetail?id=1731195978877175418 |
| 6d96f2863f27 | easylearn.baidu.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://easylearn.baidu.com/edu-page/tiangong/questiondetail?id=1732912322103810735 |
| 6fa69fee03f3 | k.sina.com.cn | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://k.sina.com.cn/article_3138941747_mbb187333033017i38.html |
| 76552a732c35 | caifuhao.eastmoney.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://caifuhao.eastmoney.com/news/20260821090522682767980 |
| 8cc538a9feb3 | easylearn.baidu.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://easylearn.baidu.com/edu-page/tiangong/questiondetail?id=1722180306057754294 |
| 9b5807a50901 | www.myzaker.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://www.myzaker.com/article/6a75bcc68e9f0942646d9904 |
| 9dd2963e9029 | caifuhao.eastmoney.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://caifuhao.eastmoney.com/news/1659116586 |
| 9e51f68c9f89 | www.php.cn | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://www.php.cn/faq/2072510.html |
| acb8cc1e4d98 | www.php.cn | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://www.php.cn/faq/2062341.html |
| b6e45f2a578f | m.toutiao.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://m.toutiao.com/a7624165826588934682/ |
| cc621e46ed2c | easylearn.baidu.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://easylearn.baidu.com/edu-page/tiangong/questiondetail?id=1709726704823465300 |
| cd294d5b372f | baike.baidu.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://baike.baidu.com/tashuo/browse/content?id=d4666d6439ad4e8d293d8cfe |
| e93d5ea0f613 | news.qq.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | https://news.qq.com/rain/a/20260806A084KK |
| fd1021096cb6 | www.myzaker.com | 未找到可用正文（可能为空页、仅脚本或需要浏览器渲染） | http://www.myzaker.com/article/5d6c803f8e9f09534b6f674b/ |

## 降级页（body 兜底）

| doc_id | host | text_chars | url |
|---|---|---:|---|
| 3c41caa58734 | www.bjwmb.gov.cn | 3480 | https://www.bjwmb.gov.cn/wmpl/10127148.html |
| 6b17f18e4910 | finance.stockstar.com | 1493 | https://finance.stockstar.com/RB2026082300011815.shtml |
| aa96f4efb050 | cj.sina.com.cn | 17 | https://cj.sina.com.cn/articles/view/7879922978/1d5ae152201901d368 |
| b0e772c52dd8 | finance.ce.cn | 191 | http://finance.ce.cn/stock/gsgdbd/202608/t20260809_3136895.shtml |
| cf2697866648 | www.163.com | 7 | https://www.163.com/v/video/VJ4312DHI.html |
| dcf1201334fe | news.bjd.com.cn | 1400 | https://news.bjd.com.cn/2026/08/07/90089853.shtml |
| fb5149044f63 | zjj.qhd.gov.cn | 721 | http://zjj.qhd.gov.cn/home/details?code=MzQ%CE%B3&id=11024 |

## 优先抽检候选

| doc_id | host | text_chars | ratio | content_root | url |
|---|---|---:|---:|---|---|
| cf2697866648 | www.163.com | 7 | 0.0588 | body | https://www.163.com/v/video/VJ4312DHI.html |
| ceeed226f24d | view.inews.qq.com | 9 | 0.0003 | body > div.not-found_notFoundWrapper__DNXcS > div.not-found_hotRankTitleWrapper__3WnFr > div.not-found_hotRankTitle__nvNKY | https://view.inews.qq.com/k/20231205A0A7YZ00 |
| c6ab75e4b3a6 | www.qlwb.com.cn | 15 | 0.0091 | body > div#__nuxt > div#nuxt-loading > div | https://www.qlwb.com.cn/videoDetail/27880644.html |
| aa96f4efb050 | cj.sina.com.cn | 17 | 0.1009 | body | https://cj.sina.com.cn/articles/view/7879922978/1d5ae152201901d368 |
| 915a0209a609 | www.anjuke.com | 19 | 0.0178 | body > div.wrapper > div.info | https://www.anjuke.com/xinfang/wx-lp252006/ |
| 16f9f38a1d8c | m.baike.com | 20 | 0.0014 | body > div > main.main > div.tip | https://m.baike.com/wikiid/4526886011186755773 |
| ac8a506ea6ec | m.baike.com | 20 | 0.0014 | body > div > main.main > div.tip | https://m.baike.com/wikiid/9015133873517104099 |
| d2c949400779 | m.baike.com | 20 | 0.0014 | body > div > main.main > div.tip | https://m.baike.com/wikiid/7511289445276500706 |
| efeb291e0806 | health.baidu.com | 22 | 0.0004 | body > div#__next > div.index_deadLinkContentArea__IrxG0 > div.index_container__ktvXm | https://health.baidu.com/m/detail/ar_2618149477636123831 |
| 36d8eb14d969 | www.sina.cn | 23 | 0.0068 | body > main.page_main > section#j_card_tnews > section#j_smartSlide > section.swiper-container | https://www.sina.cn/gc/article/ninnqpr1523069.html |
| 0965cf7c08ab | applocal.myzaker.com | 24 | 0.0424 | body > div | http://applocal.myzaker.com/news/article.php?pk=6a7458d88e9f09622124264d |
| 28e82c793cdb | news.sohu.com | 36 | 0.0145 | body > div.wrapper-box > div.area > div.err404 | http://news.sohu.com/a/1030907932_121070035 |
| 140751f2e52d | m.baidu.com | 51 | 0.0005 | body > div#__next > main#main > div.index_content__KwRj_ > div.index_healthMain__mnr20 > div.index_healthLeft__M3mi4 > div.index_healthArea__O__cV > div#aArea > div.index_article__7HJ_l > section > div#spread-fold > div | https://m.baidu.com/bh/m/detail/ar_9819638643537804340 |
| b0e772c52dd8 | finance.ce.cn | 191 | 0.0247 | body | http://finance.ce.cn/stock/gsgdbd/202608/t20260809_3136895.shtml |
| fb5149044f63 | zjj.qhd.gov.cn | 721 | 0.2973 | body | http://zjj.qhd.gov.cn/home/details?code=MzQ%CE%B3&id=11024 |
| dcf1201334fe | news.bjd.com.cn | 1400 | 0.2938 | body | https://news.bjd.com.cn/2026/08/07/90089853.shtml |
| 6b17f18e4910 | finance.stockstar.com | 1493 | 0.3252 | body | https://finance.stockstar.com/RB2026082300011815.shtml |
| 3c41caa58734 | www.bjwmb.gov.cn | 3480 | 0.2561 | body | https://www.bjwmb.gov.cn/wmpl/10127148.html |

## 文件

- `pages/`：每页简版 HTML（按 doc_id 命名）
- `index.jsonl`：每页状态、置信度、字符数、content_root、告警与 URL
- `audit.json`：机器可读汇总
