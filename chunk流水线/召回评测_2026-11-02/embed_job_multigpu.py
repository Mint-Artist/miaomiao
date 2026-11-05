"""向量推理任务（多卡版）：给任务队列用。单卡版是 embed_job.py，两者产出完全相同。
一个任务可以用一张卡，也可以用一台机器上的多张卡，推理一种或几种切分方法的全部 chunk。

与 recall_pipeline.py 第 10–11 段的计算完全相同（同样的分批、池化、fp16 存储），产出的文件格式也相同，
notebook 第 12 段以后的检索、评测可以直接读。前提：已经在 notebook 里跑完第 9 段切分（chunks/ 下有 .done）。

用法：
  单卡，一种方法（任务里只看到一张卡时 --device 用默认的 cuda:0）：
    python embed_job_multigpu.py --method md_struct --out-root /path/to/output/chunk_eval --model-path /path/to/model --loader sentence_transformers
  一台机器 8 张卡，三种方法一起算（每张卡起一个进程，各加载一份模型，所有"方法 × 分片"轮流分给各卡）：
    python embed_job_multigpu.py --method content_br md_struct html_struct --gpus all ...（其余参数相同）
    --gpus 也可以写成 0,1,2,3 只用其中几张；"all" 表示任务能看到的全部卡（会遵守任务队列设置的 CUDA_VISIBLE_DEVICES）。
  多台机器：每台提交一个任务，再加 --num-shards 机器数 --shard-id 0..机器数-1（可与 --gpus 同时用）。

断点续跑：以分片为单位。每个分片先写到临时文件，全部写完再改名，最后写 .done；任务被抢占或失败后重新提交即可，
已完成的分片自动跳过，没完成的分片从头重算。注意：同样的 --method / --num-shards / --shard-id 不要同时跑两个任务。

退出码：0 = 本任务负责的分片全部完成；1 = 配置与已有向量不一致，或某个进程出错；3 = 有分片的切分还没完成，被跳过了。
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


LOG_TAG = ""  # 多卡时每个进程的日志前面带上 [GPU x]


def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + LOG_TAG + msg, flush=True)


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


def write_meta(edir: str, meta: dict, dim):
    if dim is None:  # transformers 加载方式下，第一批编码之前不知道维度（只会发生在开头就是空分片时），下个分片再写
        return
    mpath = os.path.join(edir, "meta.json")
    tmp = f"{mpath}.tmp-{os.getpid()}"  # 多个进程写的内容相同；各用自己的临时文件名再改名，避免互相覆盖
    with open(tmp, "w") as f:
        json.dump({**meta, "dim": dim}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, mpath)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="chunk 向量推理（任务队列版，支持单机多卡）")
    ap.add_argument("--method", required=True, nargs="+", help="切分方法名，可写多个，对应 OUT_ROOT/方法名/chunks")
    ap.add_argument("--out-root", required=True, help="与 notebook 的 OUT_ROOT 相同")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--loader", default="transformers", choices=["transformers", "sentence_transformers"])
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean", "last"], help="只对 --loader transformers 生效")
    ap.add_argument("--doc-prefix", default="", help="chunk 前缀，与 notebook 的 DOC_PREFIX 相同")
    ap.add_argument("--query-prefix", default="", help="只记录到 meta.json 里，评测时用")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256, help="显存不够会自动减半重试")
    ap.add_argument("--device", default="cuda:0", help="单卡时用哪张卡；用了 --gpus 时忽略")
    ap.add_argument("--gpus", default=None, help='单机多卡："all" 或 "0,1,2,3"；每张卡起一个进程')
    ap.add_argument("--num-shards", type=int, default=1, help="多台机器（多个任务）分工时的任务总数")
    ap.add_argument("--shard-id", type=int, default=0, help="本任务的编号 0..num-shards-1")
    ap.add_argument("--block", type=int, default=200_000, help="每次送进 encode 的 chunk 数（逐块写盘）")
    ap.add_argument("--log-every", type=float, default=60, help="进度日志间隔（秒）")
    ap.add_argument("--no-trust-remote-code", action="store_true")
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)  # 内部用：多卡时子进程的 "第几个/共几个"
    return ap.parse_args(argv)


def launch_gpus(a) -> int:
    """单机多卡：每张卡起一个子进程（CUDA_VISIBLE_DEVICES 只给它一张卡），各自负责一部分"方法 × 分片"，等全部结束。"""
    import signal
    import subprocess

    visible = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    if a.gpus == "all":
        gpus = visible or [str(i) for i in range(torch.cuda.device_count())]
    else:
        idx = [g.strip() for g in a.gpus.split(",") if g.strip()]
        gpus = [visible[int(g)] for g in idx] if visible else idx  # 任务队列已限定可见卡时，编号按可见卡排
    if not gpus:
        log("没有找到可用的 GPU")
        return 1
    n = len(gpus)
    threads = max(1, (os.cpu_count() or n) // n)  # CPU 线程平分给各进程，避免分词等 CPU 计算互相抢
    log(f"单机多卡：{n} 张卡 {gpus}，方法 {a.method}，每个进程 CPU 线程 {threads}")
    procs = []
    for i, g in enumerate(gpus):
        cmd = [sys.executable, os.path.abspath(__file__), "--method", *a.method, "--out-root", a.out_root,
               "--model-path", a.model_path, "--loader", a.loader, "--pooling", a.pooling,
               "--doc-prefix", a.doc_prefix, "--query-prefix", a.query_prefix, "--max-length", str(a.max_length),
               "--batch-size", str(a.batch_size), "--block", str(a.block), "--log-every", str(a.log_every),
               "--num-shards", str(a.num_shards), "--shard-id", str(a.shard_id), "--_worker", f"{i}/{n}",
               "--device", "cuda:0" if str(a.device).startswith("cuda") else a.device]
        if a.no_trust_remote_code:
            cmd.append("--no-trust-remote-code")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        procs.append(subprocess.Popen(cmd, env=env))

    def stop(signum, frame):  # 任务被取消 / 抢占时，把子进程一起停掉
        log(f"收到信号 {signum}，停止全部子进程")
        for p in procs:
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    codes = [p.wait() for p in procs]
    log(f"全部子进程结束，退出码 {codes}")
    if all(c == 0 for c in codes):
        return 0
    if all(c in (0, 3) for c in codes):
        return 3
    return 1


def main(argv=None):
    global LOG_TAG
    a = parse_args(argv)
    if a.gpus and not a._worker:
        return launch_gpus(a)

    # 本进程在全部"方法 × 分片"里负责的那一份：多台机器（num_shards）× 每台的卡数（worker）
    wi, wn = (int(x) for x in a._worker.split("/")) if a._worker else (0, 1)
    k, n_all = a.shard_id * wn + wi, a.num_shards * wn
    if a._worker:
        LOG_TAG = f"[GPU {wi}] "
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())))

    meta = {"model_path": a.model_path, "loader": a.loader, "pooling": a.pooling, "max_length": a.max_length,
            "doc_prefix": a.doc_prefix, "query_prefix": a.query_prefix, "dtype": "float16", "normalized": True}
    units, not_chunked, chunks_of = [], 0, {}
    for method in a.method:
        cdir = os.path.join(a.out_root, method, "chunks")
        edir = os.path.join(a.out_root, method, "emb")
        if not os.path.isdir(cdir):
            log(f"找不到 {cdir}：先在 notebook 里跑完第 9 段切分")
            return 3
        os.makedirs(edir, exist_ok=True)
        mpath = os.path.join(edir, "meta.json")
        if os.path.exists(mpath):  # 新旧向量必须来自同一模型、同一 chunk 前缀，否则不能放进同一个索引
            old = json.load(open(mpath))
            diff = {f: (old.get(f), meta[f]) for f in ("model_path", "loader", "pooling", "max_length", "doc_prefix")
                    if old.get(f) != meta[f]}
            if diff:
                log(f"[{method}] 与已有向量的配置不一致（旧值, 新值）：{diff}；确需更换模型请换一个 out-root 重跑")
                return 1
        names = sorted(os.path.basename(p)[: -len(".jsonl")] for p in glob.glob(os.path.join(cdir, "*.jsonl")))
        for n in names:
            if not os.path.exists(os.path.join(cdir, n + ".done")):
                not_chunked += 1
            else:
                units.append((method, n))
    # 按 chunk 数分配：从大到小，每个分片交给当前工作量最少的进程。各进程读到的 .done 相同，算出的分配也相同，互不重叠
    for u in units:
        chunks_of[u] = json.load(open(os.path.join(a.out_root, u[0], "chunks", u[1] + ".done")))["chunks"]
    load, owner = [0] * n_all, {}
    for u in sorted(units, key=lambda u: (-chunks_of[u], u)):
        w = min(range(n_all), key=lambda i: (load[i], i))
        owner[u] = w
        load[w] += chunks_of[u]
    mine = [u for u in units if owner[u] == k]
    todo = [(m, n) for m, n in mine if not os.path.exists(os.path.join(a.out_root, m, "emb", n + ".done"))]
    total = sum(chunks_of[u] for u in todo)
    log(f"方法 {a.method}，第 {k}/{n_all} 份：分片 {len(mine)} 个，待推理 {len(todo)} 个，共 {total} 个 chunk"
        + (f"；另有 {not_chunked} 个分片切分未完成，已跳过" if not_chunked else ""))
    if not todo:
        return 3 if not_chunked else 0

    t0 = time.time()
    enc = Encoder(a.model_path, a.device, a.loader, a.pooling, a.max_length, a.batch_size,
                  trust_remote_code=not a.no_trust_remote_code)
    gpu = torch.cuda.get_device_name(a.device) if str(a.device).startswith("cuda") else "cpu"
    log(f"模型加载完成（{gpu}，用时 {time.time() - t0:.0f}s），batch_size={a.batch_size}")

    pbar = Progress(total, "进度", a.log_every)
    for j, (method, n) in enumerate(todo):
        cdir = os.path.join(a.out_root, method, "chunks")
        edir = os.path.join(a.out_root, method, "emb")
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
        write_meta(edir, meta, enc.dim)  # 先于 .done 写：只要有完成的分片，就一定有 meta.json（中途被取消也不会缺）
        open(os.path.join(edir, n + ".done"), "w").close()
        log(f"[{method}] 分片 {n} 完成（{j + 1}/{len(todo)}，{len(texts)} 个 chunk）")

    el = time.time() - pbar.t0
    log(f"全部完成：{total} 个 chunk，用时 {fmt_secs(el)}，平均 {total / max(el, 1e-9):.0f} 条/秒")
    return 3 if not_chunked else 0


if __name__ == "__main__":
    sys.exit(main())
