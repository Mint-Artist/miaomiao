"""单任务向量推理：给任务队列用，一个任务（一张卡）推理一种切分方法的全部 chunk。

与 recall_pipeline.py 第 10–11 段的计算完全相同（同样的分批、池化、fp16 存储），产出的文件格式也相同，
notebook 第 12 段以后的检索、评测可以直接读。前提：已经在 notebook 里跑完第 9 段切分（chunks/ 下有 .done）。

用法（每种方法提交一个任务；任务里只看到一张卡时 --device 用默认的 cuda:0）：
  python embed_job.py --method content_br  --out-root /path/to/output/chunk_eval --model-path /path/to/model --loader sentence_transformers
  python embed_job.py --method md_struct   ...（其余参数相同）
  python embed_job.py --method html_struct ...
一种方法想再拆给多张卡：同一方法提交多个任务，分别加 --num-shards N --shard-id 0..N-1（按分片轮流分配，互不重叠）。

断点续跑：以分片为单位。每个分片先写到临时文件，全部写完再改名，最后写 .done；任务被抢占或失败后重新提交即可，
已完成的分片自动跳过，没完成的分片从头重算。注意：同一方法、同一 shard-id 不要同时跑两个任务。

退出码：0 = 本任务负责的分片全部完成；1 = 配置与已有向量不一致；3 = 有分片的切分还没完成，被跳过了。
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 减少显存碎片导致的误报 OOM；须在 import torch 之前

from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def fmt_secs(s: float) -> str:
    if s == float("inf"):
        return "—"
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Progress:
    """任务队列的日志一般不是终端，tqdm 的进度条会刷成成千上万行；这里每隔 every 秒打印一行。"""

    def __init__(self, total: int, desc: str, every: float = 60):
        self.total, self.desc, self.every = total, desc, every
        self.n, self.t0 = 0, time.time()
        self.last = self.t0

    def update(self, k: int):
        self.n += k
        now = time.time()
        if now - self.last >= self.every or self.n >= self.total:
            self.last = now
            el = now - self.t0
            sp = self.n / el if el > 0 else 0.0
            eta = (self.total - self.n) / sp if sp > 0 else float("inf")
            log(f"{self.desc} {self.n}/{self.total}（{100 * self.n / max(self.total, 1):.1f}%） "
                f"{sp:.0f} 条/秒  已用 {fmt_secs(el)}  预计剩余 {fmt_secs(eta)}")


# ---- 以下 Encoder 与 read_chunks 与 recall_pipeline.py 第 10–11 段一致 ----

class Encoder:
    def __init__(self, path, device, loader, pooling, max_length, batch_size,
                 dtype=torch.float16, normalize=True, trust_remote_code=True):
        self.device, self.pooling, self.max_length, self.normalize, self.loader = device, pooling, max_length, normalize, loader
        self.batch_size = batch_size
        if loader == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(path, device=device, trust_remote_code=trust_remote_code)
            self.model.max_seq_length = max_length
            if str(device).startswith("cuda"):
                self.model.half()
            self.dim = self.model.get_sentence_embedding_dimension()
        else:
            from transformers import AutoModel, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
            dt = dtype if str(device).startswith("cuda") else torch.float32
            self.model = AutoModel.from_pretrained(path, torch_dtype=dt, trust_remote_code=trust_remote_code).to(device).eval()
            self.dim = None

    @torch.inference_mode()
    def _batch(self, texts: List[str]) -> np.ndarray:
        if self.loader == "sentence_transformers":
            v = self.model.encode(texts, batch_size=len(texts), convert_to_tensor=True, normalize_embeddings=self.normalize,
                                  show_progress_bar=False)
            return v.float().cpu().numpy().astype(np.float16)
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        h = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"]
        if self.pooling == "cls":
            v = h[:, 0]
        elif self.pooling == "mean":
            m = mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.pooling == "last":
            if self.tok.padding_side == "left":
                v = h[:, -1]
            else:
                v = h[torch.arange(h.size(0), device=h.device), mask.sum(1) - 1]
        else:
            raise ValueError(self.pooling)
        v = v.float()
        if self.normalize:
            v = F.normalize(v, dim=-1)
        return v.cpu().numpy().astype(np.float16)

    def encode(self, texts: List[str], prefix: str = "", batch_size: int = None, pbar=None) -> np.ndarray:
        """按长度从长到短排序后分批（减少 padding，且最长的批先跑，显存不够会立刻暴露），结果按原顺序返回。"""
        n = len(texts)
        order = np.argsort([-len(t) for t in texts], kind="stable")
        out, i, bs = None, 0, batch_size or self.batch_size
        while i < n:
            idx = order[i:i + bs]
            try:
                v = self._batch([prefix + texts[j] for j in idx])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                log(f"显存不足，batch 降到 {bs}")
                continue
            if out is None:
                out = np.empty((n, v.shape[1]), dtype=np.float16)
                self.dim = v.shape[1]
            out[idx] = v
            i += len(idx)
            if pbar is not None:
                pbar.update(len(idx))
        return out if out is not None else np.zeros((0, self.dim or 0), dtype=np.float16)


def read_chunks(chunk_jsonl: str):
    texts, offsets, urls = [], [0], []
    with open(chunk_jsonl, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.extend(d["chunks"])
            offsets.append(len(texts))
            urls.append(d["url"])
    return texts, np.asarray(offsets, dtype=np.int64), urls


# ---- 任务主体 ----

def _tmp(path: str) -> str:
    return path + ".tmp.npy"  # 固定名字：任务重跑时覆盖上次没写完的临时文件，不会越积越多


def save_npy(path: str, arr: np.ndarray):
    np.save(_tmp(path), arr)
    os.replace(_tmp(path), path)


def main():
    ap = argparse.ArgumentParser(description="一种切分方法的 chunk 向量推理（任务队列版）")
    ap.add_argument("--method", required=True, help="切分方法名，对应 OUT_ROOT/方法名/chunks")
    ap.add_argument("--out-root", required=True, help="与 notebook 的 OUT_ROOT 相同")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--loader", default="transformers", choices=["transformers", "sentence_transformers"])
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean", "last"], help="只对 --loader transformers 生效")
    ap.add_argument("--doc-prefix", default="", help="chunk 前缀，与 notebook 的 DOC_PREFIX 相同")
    ap.add_argument("--query-prefix", default="", help="只记录到 meta.json 里，评测时用")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256, help="显存不够会自动减半重试")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--block", type=int, default=200_000, help="每次送进 encode 的 chunk 数（逐块写盘）")
    ap.add_argument("--log-every", type=float, default=60, help="进度日志间隔（秒）")
    ap.add_argument("--no-trust-remote-code", action="store_true")
    a = ap.parse_args()

    method = a.method
    cdir = os.path.join(a.out_root, method, "chunks")
    edir = os.path.join(a.out_root, method, "emb")
    if not os.path.isdir(cdir):
        log(f"找不到 {cdir}：先在 notebook 里跑完第 9 段切分")
        return 3
    os.makedirs(edir, exist_ok=True)

    names = sorted(os.path.basename(p)[: -len(".jsonl")] for p in glob.glob(os.path.join(cdir, "*.jsonl")))
    names = [n for k, n in enumerate(names) if k % a.num_shards == a.shard_id]
    not_chunked = [n for n in names if not os.path.exists(os.path.join(cdir, n + ".done"))]
    names = [n for n in names if n not in not_chunked]
    todo = [n for n in names if not os.path.exists(os.path.join(edir, n + ".done"))]

    meta = {"model_path": a.model_path, "loader": a.loader, "pooling": a.pooling, "max_length": a.max_length,
            "doc_prefix": a.doc_prefix, "query_prefix": a.query_prefix, "dtype": "float16", "normalized": True}
    mpath = os.path.join(edir, "meta.json")
    if os.path.exists(mpath):  # 新旧向量必须来自同一模型、同一 chunk 前缀，否则不能放进同一个索引
        old = json.load(open(mpath))
        diff = {k: (old.get(k), meta[k]) for k in ("model_path", "loader", "pooling", "max_length", "doc_prefix")
                if old.get(k) != meta[k]}
        if diff:
            log(f"[{method}] 与已有向量的配置不一致（旧值, 新值）：{diff}；确需更换模型请换一个 out-root 重跑")
            return 1

    total = sum(json.load(open(os.path.join(cdir, n + ".done")))["chunks"] for n in todo)
    log(f"[{method}] shard {a.shard_id}/{a.num_shards}：分片 {len(names)} 个，待推理 {len(todo)} 个，共 {total} 个 chunk"
        + (f"；另有 {len(not_chunked)} 个分片切分未完成，已跳过" if not_chunked else ""))
    if not todo:
        return 3 if not_chunked else 0

    t0 = time.time()
    enc = Encoder(a.model_path, a.device, a.loader, a.pooling, a.max_length, a.batch_size,
                  trust_remote_code=not a.no_trust_remote_code)
    gpu = torch.cuda.get_device_name(a.device) if str(a.device).startswith("cuda") else "cpu"
    log(f"模型加载完成（{gpu}，用时 {time.time() - t0:.0f}s），batch_size={a.batch_size}")

    pbar = Progress(total, f"[{method}]", a.log_every)
    for k, n in enumerate(todo):
        texts, offsets, _ = read_chunks(os.path.join(cdir, n + ".jsonl"))
        path = os.path.join(edir, n + ".npy")
        mm = None
        for s in range(0, len(texts), a.block):  # 分块推理、逐块写盘，内存占用与分片大小无关
            v = enc.encode(texts[s:s + a.block], prefix=a.doc_prefix, pbar=pbar)
            if mm is None:
                mm = np.lib.format.open_memmap(_tmp(path), mode="w+", dtype=np.float16, shape=(len(texts), v.shape[1]))
            mm[s:s + len(v)] = v
        if mm is None:  # 空分片
            save_npy(path, np.zeros((0, enc.dim or 1), dtype=np.float16))
        else:
            mm.flush()
            del mm
            os.replace(_tmp(path), path)
        save_npy(os.path.join(edir, n + ".offsets.npy"), offsets)
        save_npy(os.path.join(edir, n + ".lens.npy"), np.asarray([len(t) for t in texts], dtype=np.int32))
        open(os.path.join(edir, n + ".done"), "w").close()
        log(f"[{method}] 分片 {n} 完成（{k + 1}/{len(todo)}，{len(texts)} 个 chunk）")

    meta["dim"] = enc.dim
    with open(mpath + ".tmp", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    os.replace(mpath + ".tmp", mpath)
    el = time.time() - pbar.t0
    log(f"[{method}] 全部完成：{total} 个 chunk，用时 {fmt_secs(el)}，平均 {total / max(el, 1e-9):.0f} 条/秒")
    return 3 if not_chunked else 0


if __name__ == "__main__":
    sys.exit(main())
