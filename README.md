# miaomiao

## SELECT / BidirLM

- `sequence_BIO/`：从原始文本和 deletion-only 清洗文本生成 BIO 训练数据。
- `bidirlm_BIO_finetune/`：使用本地 BidirLM-0.6B 进行单卡或多卡 SELECT 风格微调。

完整训练命令见 `bidirlm_BIO_finetune/README.md`。
