# npv：网页权威度打分（PageValueScore）的分析与 Python 实验环境

原始 Spark 作业的 Java 源码与扫描件不在本仓库中，这里只包含分析文档与 Python 代码。

- `claude/PageValueScore解读报告.md`：模型解读、问题清单（P0/P1/P2）与改进建议。
- `claude/立即修复变更说明.md`：七条立即修复项的逐条说明（修改后的 Java 不在本仓库）。
- `claude/npv_exact/`：原始逻辑的逐行 Python 复刻，**保留全部缺陷**并模拟 Java 数值语义，用于与线上输出对齐。
- `claude/与原Java的差异记录.md`：三个 Python 版本相对原始逻辑的全部差异与复现不一致时的排查顺序。
- `claude/npv_py/`：打分逻辑的 Python 复刻（修复版），冻结基线；含 CLI、合成样例、与原作业输出的比对工具、单元测试。
- `claude/npv_lab/`：实验层：实验特征、逐步解释、结构诊断、离线评估（偏序一致率 / 误压率 / 配对检验）、偏序权重拟合、实验记录。
- `claude/权威度调研/`：业界与学术界权威度方案调研、行动清单与待对齐问题。

快速验证（纯标准库，Python 3.8+）：

```bash
cd npv/claude/npv_py && python -m unittest discover -s tests
cd ../npv_lab && python -m unittest discover -s tests
cd ../npv_exact && python -m unittest test_exact
```
