# 内部流水线代码规范

本文档记录 `sequence_BIO` 和 `bidirlm_BIO_finetune` 当前已确认的内部流水线检查规则。

## 1. 禁止使用 `print`

使用标准日志模块输出运行信息。

```python
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.info("Processing completed")
```

## 2. 字典读取使用 `get`

读取字典字段时禁止直接使用 `value["key"]`，应使用 `get` 并提供类型合适的默认值。

```python
content = record.get("content", "")
labels = record.get("labels", [])
metadata = record.get("metadata", {})
```

该规则针对字典读取；列表、元组和张量仍可按索引访问。字典赋值不受影响。

## 3. 使用完整包名导入

禁止点号相对导入，应使用完整包名。

```python
# 禁止
from .constants import BIO_LABELS

# 使用
from sequence_BIO.constants import BIO_LABELS
```

程序应从仓库根目录以模块方式运行，例如：

```bash
python -m bidirlm_BIO_finetune.train
```

## 4. 推导式和生成器仅用于简单表达式

包含多个条件、嵌套分支或复杂转换时，应改用普通循环。

```python
selected = []
for item in values:
    if not is_valid(item):
        continue
    if should_skip(item):
        continue
    selected.append(item)
```

## 5. 拆分复杂条件

条件语句和循环语句中不能集中包含过多条件。优先采用以下方式：

- 提前返回；
- 使用 `continue` 跳过无效数据；
- 提取职责单一的判断函数；
- 使用语义明确的布尔变量。

## 6. 禁止使用可变对象作为默认参数

不得使用列表、字典、集合或其他可变对象作为函数默认值。

```python
# 禁止
def process(options: dict = {}) -> None:
    ...

# 使用
def process(options: dict | None = None) -> None:
    resolved_options = {} if options is None else options
```

普通可变对象可以作为运行时实参传入；禁止的是在函数定义中复用可变默认对象。

## 7. 控制圈复杂度和嵌套深度

函数应保持单一职责，避免过深的 `if`、`for` 和 `try` 嵌套。复杂流程应拆分为命名明确的小函数，并尽量采用提前返回。

## 8. 禁止魔鬼数字和重复字面量

业务含义明确的数值、标签和重复字符串应集中定义为命名常量。

```python
INSIDE_LABEL_ID = 2

if label_id == INSIDE_LABEL_ID:
    ...
```

## 提交前检查

提交代码前至少执行：

```bash
python -m ruff check --select E4,E7,E9,F,I bidirlm_BIO_finetune sequence_BIO
python -m unittest discover -s sequence_BIO/tests -t . -v
python -m unittest discover -s bidirlm_BIO_finetune/tests -t . -v
```

