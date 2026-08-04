# ProX 第一阶段：JSONL 文档价值过滤

本目录用于在单张 32GB GPU 上，通过 Hugging Face Transformers 运行 ProX 文档级模型，对已经抽取成纯文本的网页文档标记：

- `keep`：保留，继续进入后续 Chunk 精炼；
- `drop`：整篇文档不适合作为预训练语料；
- `unknown`：模型输出无法严格解析，默认保留原文并等待审计。

这里判断的是“是否适合作为通用语言模型预训练语料”，不是事实正确性、公司业务价值、敏感信息、版权或重复检测。

## 推荐使用方式：先从 Notebook 开始

如果你需要在公司环境逐步排查 CUDA、依赖、模型目录、prompt 或输出问题，首先打开：

- [`prox_doc_filter_notebook.ipynb`](prox_doc_filter_notebook.ipynb)

Notebook 没有把流程封装进类或大型函数，而是依次展示：

1. 修改本地路径和文本键；
2. 检查 PyTorch/CUDA；
3. 单独加载 tokenizer；
4. 单独加载模型；
5. 构造一条手写文本；
6. 查看完整 Llama-2 prompt；
7. 查看 token 数和截断情况；
8. 生成并打印模型原始程序；
9. 自己修改 `keep/drop/unknown` 解析规则；
10. 只读取少量 JSONL；
11. 逐条推理并人工检查；
12. 确认后才写出结果。

建议第一轮保持 `MAX_RECORDS=20`，不要直接运行全量。目录中的 `run_doc_filter.py` 是逻辑确认后的可选批处理参考，不是必须使用的入口。

## 1. 模型介绍与下载地址

本阶段使用：

- 模型主页：[gair-prox/web-doc-refining-lm](https://huggingface.co/gair-prox/web-doc-refining-lm)
- ProX 代码：[GAIR-NLP/ProX](https://github.com/GAIR-NLP/ProX)
- 论文：[Programming Every Example](https://arxiv.org/abs/2409.17115)

模型信息：

| 项目 | 内容 |
|---|---|
| 架构 | `LlamaForCausalLM` |
| 参数量 | 354,284,800，约 0.35B |
| 权重文件 | FP32 Safetensors，约 1.42GB |
| 最大上下文 | 2048 token |
| 主要用途 | 英文通用网页文档级 keep/drop 过滤 |
| 许可证 | Apache-2.0 |

单卡 32GB 显存非常充足。脚本会优先使用 BF16，不支持 BF16 时使用 FP16；CPU 模式使用 FP32。vLLM 不是必需依赖。

在有网络的机器下载模型：

```bash
cd prox_doc_filter
python download_model.py \
  --output-dir models/web-doc-refining-lm
```

然后将整个 `models/web-doc-refining-lm/` 复制到公司环境。也可以直接从模型主页手工下载全部文件。

## 2. 环境准备

建议使用 Python 3.10。先根据公司 CUDA 版本安装匹配的 PyTorch，确认：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

输出中的 CUDA 可用状态应为：

```text
True
```

再安装其余依赖：

```bash
pip install -r requirements.txt
```

公司离线环境需要提前下载 PyTorch 和 `requirements.txt` 中所有依赖的 wheel，或使用公司内部 PyPI/Conda 镜像。

## 3. 输入 JSONL 格式

每个物理行必须是一个完整 JSON 对象，例如：

```jsonl
{"id":"doc-001","url":"https://example.com/1","content":"Title\n\nArticle body.\n\nCopyright 2026"}
{"id":"doc-002","url":"https://example.com/2","content":"Home\nLogin\nCookie settings"}
```

要求：

1. 文本已经从 HTML 抽取为纯文本；
2. 标题、段落、菜单、页脚之间保留合理的 `\n`；
3. JSONL 中换行以转义形式 `\n` 存储，`json.loads` 后会恢复成真正换行；
4. 第一阶段不需要添加 `[000]` 行号或 `[doc]` 包装；
5. 文本字段名可配置，默认是 `content`；
6. 嵌套字段可以写成 `data.cleaned_text`。

先执行无需 GPU 的输入检查：

```bash
python validate_input.py \
  --input example_input.jsonl \
  --text-key content
```

存在非法 JSON、缺少字段或字段不是字符串时，命令返回非零退出码。

## 4. 在 Jupyter 中逐步运行

如果环境已经安装 Jupyter：

```bash
cd prox_doc_filter
jupyter lab prox_doc_filter_notebook.ipynb
```

也可以在已有的 Jupyter 服务中打开该文件。公司离线环境应在第一个配置 cell 中设置：

```python
MODEL_PATH = "/models/prox/web-doc-refining-lm"
LOCAL_FILES_ONLY = True
INPUT_PATH = Path("/data/input/sample.jsonl")
TEXT_KEY = "content"
```

每次只执行一个 cell；一旦出错，可以直接查看当前的 `tokenizer`、`inputs`、`program` 等变量，而不需要穿过完整 CLI 或类封装。

## 5. 可选：在线批处理脚本

如果运行环境可以连接 Hugging Face：

```bash
python run_doc_filter.py \
  --input example_input.jsonl \
  --output output/example_output.jsonl \
  --text-key content \
  --model gair-prox/web-doc-refining-lm \
  --device cuda \
  --batch-size 8
```

首次运行会下载模型。

## 6. 可选：公司离线批处理脚本

设置完全离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

使用复制进公司的本地模型目录：

```bash
python run_doc_filter.py \
  --input /data/input/part-00000.jsonl \
  --output /data/output/part-00000.jsonl \
  --text-key content \
  --model /models/prox/web-doc-refining-lm \
  --local-files-only \
  --device cuda \
  --dtype auto \
  --batch-size 8
```

输入和输出也支持 `.jsonl.gz`。脚本不允许默认覆盖已有输出；确实需要覆盖时显式添加：

```text
--overwrite
```

## 7. 输出字段

脚本保留原 JSON 的全部字段，并添加：

| 字段 | 含义 |
|---|---|
| `prox_doc_program` | 模型未经修改的原始输出 |
| `prox_doc_decision` | `keep`、`drop` 或 `unknown` |
| `prox_doc_text` | `drop` 时为空；其他情况为规范化后的原文 |
| `prox_doc_truncated` | 输入是否因 2048 token 上下文限制被截断 |
| `prox_doc_error` | 推理错误；正常时为 `null` |

示例：

```jsonl
{"id":"doc-001","content":"Title\n\nArticle body.","prox_doc_program":"keep","prox_doc_decision":"keep","prox_doc_text":"Title\n\nArticle body.","prox_doc_truncated":false,"prox_doc_error":null}
{"id":"doc-002","content":"Home\nLogin","prox_doc_program":"drop","prox_doc_decision":"drop","prox_doc_text":"","prox_doc_truncated":false,"prox_doc_error":null}
```

脚本结束时会打印总数、keep/drop/unknown、截断和错误数量。

## 8. 解析策略

默认使用保守的严格策略：

```text
keep / keep.  -> keep
drop / drop!  -> drop
其他输出      -> unknown，并保留原文
```

这适合第一轮验证，能看清模型实际输出分布。若要复现 ProX 原仓库逻辑，可添加：

```text
--parse-mode repo-compatible
```

原仓库逻辑是：输出中包含 `drop` 就丢弃，否则保留。该模式可能把 `do not drop` 误判为 drop，因此不建议作为第一轮默认值。

## 9. 长文档处理

模型最大上下文为 2048 token。脚本会为最多 32 个输出 token 预留空间，对超长 prompt 从右侧截断，并设置：

```json
{"prox_doc_truncated": true}
```

因此长文档的第一阶段判断主要基于开头内容。第一轮测试应单独统计和抽查 `prox_doc_truncated=true` 的样本，不要直接把这类结果用于不可恢复的删除。

## 10. 推荐上线顺序

1. 先运行 `validate_input.py`；
2. 使用 100 条数据验证环境和输出格式；
3. 使用 500～2000 条分层样本人工检查；
4. 统计 `keep/drop/unknown/truncated/error`；
5. 重点检查高质量正文被误判为 `drop` 的情况；
6. 第一轮只增加标签，不从原始数据中物理删除记录；
7. 确认文档级效果后，再开发第二阶段 `web-chunk-refining-lm` 行级清洗。

公开模型主要面向英文网页语料。中文、多语言、OCR、代码、表格或行业专有文本必须分组评估，不能直接假设效果等同于英文通用网页。

## 11. 测试

核心逻辑测试不需要下载模型：

```bash
python -m unittest discover -s tests -v
```

测试覆盖严格/原仓库兼容解析、嵌套文本字段、换行规范化和 Llama-2 prompt 格式。
