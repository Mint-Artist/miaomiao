#!/usr/bin/env python3
"""agent 唯一能调用的评估入口。只接受 --hypothesis 与 --desc 两个参数。

顺序：登记实验卡 -> 静态检查 -> 打分子进程（只看快照、sandbox、train）-> 硬约束
      -> 第一道关（train 交叉验证）-> 第二道关（dev：Ladder 门槛 + 护栏）-> git 提交或回退 -> 记录。
保留与回退由本脚本执行，agent 不碰 git。不保留时 dev 只给粗档，不给精确数字。
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HARNESS_DIR, ROOT, load_harness_config, load_sandbox_config, read_grades,  # noqa: E402
                    read_json, read_pairs, read_scores, sha256_file, site_of, write_json)
from lint import lint_sandbox  # noqa: E402
from metrics import complexity, constraint_checks, grade_metrics, paired_compare, strict_pairwise, subset_accuracy  # noqa: E402
import gate  # noqa: E402

RUNS = os.path.join(ROOT, "runs")
BEST = os.path.join(RUNS, "best")
SANDBOX = os.path.join(ROOT, "sandbox")
SNAPSHOT = os.path.join(HARNESS_DIR, "snapshot")
TRAIN = os.path.join(HARNESS_DIR, "labels", "train")
DEV = os.path.join(HARNESS_DIR, "labels", "dev")
STATE = os.path.join(HARNESS_DIR, "state.json")
RESULTS = os.path.join(ROOT, "results.tsv")
COLUMNS = ["run_id", "time", "commit", "status", "stage", "dev_acc", "dev_delta", "threshold", "cv_acc",
           "train_acc", "hard_acc", "new_site_acc", "buried_good_rate", "top_bad_rate", "score_ties", "missing",
           "complexity", "dev_budget_used", "reason", "hypothesis", "desc"]
GIT_ID = ["-c", "user.name=npv_ar", "-c", "user.email=npv_ar@localhost"]


def git(*args, check=True):
    return subprocess.run(["git", *GIT_ID, *args], cwd=ROOT, capture_output=True, text=True, check=check)


def git_head():
    r = git("rev-parse", "--short", "HEAD", check=False)
    return r.stdout.strip() if r.returncode == 0 else "none"


def revert_sandbox():
    if git_head() == "none":
        return
    git("checkout", "--", "sandbox")
    git("clean", "-fdq", "sandbox")


def commit_all(msg):
    git("add", "-A")
    git("commit", "-q", "--allow-empty", "-m", msg)
    return git_head()


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else f"{v:.4f}"
    return str(v).replace("\t", " ").replace("\n", " ")


def append_results(row):
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", encoding="utf-8") as f:
        if new:
            f.write("\t".join(COLUMNS) + "\n")
        f.write("\t".join(fmt(row.get(c)) for c in COLUMNS) + "\n")


def tail(s, n=1200):
    return s if len(s) <= n else "...\n" + s[-n:]


def main(argv=None):
    p = argparse.ArgumentParser(description="运行一次实验评估（冻结脚本）", add_help=True)
    p.add_argument("--hypothesis", required=True, help="本次改动的假设与预期影响")
    p.add_argument("--desc", required=True, help="一句话描述改动")
    a = p.parse_args(argv)
    if not a.hypothesis.strip() or not a.desc.strip():
        p.error("--hypothesis 与 --desc 不能为空")

    hcfg = load_harness_config()
    state = read_json(STATE, {"n_runs": 0, "best": None, "dev_budget_used": 0, "promoted": None})
    baseline = state["best"] is None
    best = state["best"] or {}
    run_id = f"{state['n_runs'] + 1:04d}"
    run_dir = os.path.join(RUNS, run_id)
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    row = {"run_id": run_id, "time": datetime.datetime.now().isoformat(timespec="seconds"),
           "hypothesis": a.hypothesis.strip(), "desc": a.desc.strip(), "dev_budget_used": state["dev_budget_used"]}
    write_json(os.path.join(run_dir, "card.json"), {  # 实验卡：评估之前写下，之后不改
        "run_id": run_id, "time": row["time"], "hypothesis": row["hypothesis"], "desc": row["desc"],
        "parent_commit": git_head(),
        "sandbox_sha": {f: sha256_file(os.path.join(SANDBOX, f)) for f in ("features.py", "config.json")},
    })

    def finish(status, stage, reason, exit_code=0):
        if status != "keep":
            revert_sandbox()
        state["n_runs"] = int(run_id)
        if status == "keep":
            os.makedirs(BEST, exist_ok=True)
            for fn in ("npv_ori.tsv", "weights.json", "cv.json", "stats.json"):
                src = os.path.join(run_dir, fn)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(BEST, fn))
        # 两次提交：先提交代码与实验卡拿到哈希，再把哈希写进记录并提交记录，保证工作区干净
        commit = commit_all(f"{status} #{run_id}: {row['desc'][:72]}")
        if status == "keep":
            state["best"]["commit"] = commit
        row.update(status=status, stage=stage, reason=reason, commit=commit, dev_budget_used=state["dev_budget_used"])
        write_json(STATE, state)
        write_json(os.path.join(run_dir, "eval.json"), {k: row.get(k) for k in COLUMNS if k not in ("hypothesis", "desc")})
        append_results(row)
        commit_all(f"log #{run_id}")
        parts = [f"status={status}", f"#{run_id}", f"stage={stage}"]
        if status == "keep":
            parts.append(f"dev_acc={fmt(row.get('dev_acc'))}" + (f"({row['dev_delta']:+.4f}, 门槛 {row['threshold']:.4f})" if row.get("dev_delta") is not None else ""))
            for k in ("cv_acc", "hard_acc", "new_site_acc", "buried_good_rate", "top_bad_rate"):
                if row.get(k) is not None:
                    parts.append(f"{k}={fmt(row[k])}")
        else:
            if row.get("dev_acc") is not None:
                parts.append(f"dev={row['dev_acc']}")
            if row.get("cv_acc") is not None:
                parts.append(f"cv_acc={fmt(row['cv_acc'])}" + (f"({row['cv_acc'] - best['cv_accuracy']:+.4f})" if best.get("cv_accuracy") else ""))
        parts.append(f"complexity={row.get('complexity')}" + (f"({row['complexity'] - best['complexity']:+d})" if best.get("complexity") is not None and row.get("complexity") is not None else ""))
        parts.append(f"budget={state['dev_budget_used']}/{hcfg['dev_budget']}")
        parts.append(f"reason={reason}")
        parts.append(f"took={time.time() - t0:.1f}s")
        print("  ".join(parts))
        sys.exit(exit_code)

    # 1. 静态检查 + 笔记泄漏检查
    hard, review = lint_sandbox(SANDBOX, hcfg["domain_literal_limit"], hcfg["list_lines_limit"])
    train_urls = set()
    for pr in read_pairs(os.path.join(TRAIN, "pairs.tsv")):
        train_urls.update((pr["url_a"], pr["url_b"]))
    dev_only = set()
    for pr in read_pairs(os.path.join(DEV, "pairs.tsv")):
        dev_only.update(u for u in (pr["url_a"], pr["url_b"]) if u not in train_urls)
    for fn in ("journal.md", "ideas.md"):
        fp = os.path.join(ROOT, fn)
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as f:
                text = f.read()
            if any(u in text for u in dev_only):
                hard.append(f"{fn} 中出现了只在 dev 里才有的 URL")
    if hard:
        print("\n".join("LINT: " + h for h in hard), file=sys.stderr)
        for fn in ("journal.md", "ideas.md"):
            if any(h.startswith(fn) for h in hard) and git_head() != "none":
                git("checkout", "--", fn, check=False)  # 泄漏的笔记不进提交
        finish("lint_fail", "lint", "; ".join(hard)[:200], exit_code=1)

    # 2. 打分子进程
    cmd = list(hcfg.get("score_command_prefix", [])) + [
        sys.executable, os.path.join(HARNESS_DIR, "score.py"), "--sandbox", SANDBOX, "--snapshot", SNAPSHOT,
        "--train-pairs", os.path.join(TRAIN, "pairs.tsv"), "--out", run_dir, "--cv", str(hcfg["cv_folds"])]
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8", "HOME": run_dir}
    try:
        r = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, timeout=hcfg["score_timeout_sec"])
    except subprocess.TimeoutExpired:
        finish("crash", "score", f"打分超过 {hcfg['score_timeout_sec']} 秒被终止", exit_code=1)
    if r.returncode != 0:
        with open(os.path.join(run_dir, "score_stderr.txt"), "w", encoding="utf-8") as f:
            f.write(r.stderr)
        print(tail(r.stderr), file=sys.stderr)
        finish("crash", "score", (r.stderr.strip().splitlines() or ["打分失败"])[-1][:200], exit_code=1)
    stats = read_json(os.path.join(run_dir, "stats.json"))
    cv = read_json(os.path.join(run_dir, "cv.json"))
    weights = read_json(os.path.join(run_dir, "weights.json"))
    cfg, _fit = load_sandbox_config(os.path.join(SANDBOX, "config.json"))
    cx = complexity(os.path.join(SANDBOX, "features.py"), weights["fea_weight"], len(cfg.pc_rules))
    row.update(cv_acc=cv.get("mean"), train_acc=stats["train_accuracy"], complexity=cx["total"])

    # 3. 硬约束
    scores = read_scores(os.path.join(run_dir, "npv_ori.tsv"))
    cons = constraint_checks(scores)
    if stats["baseline_feature_mismatch"] > 0:
        finish("discard", "constraints", f"前 6 个基线特征被改动（{stats['baseline_feature_mismatch']} 行不一致）")
    if not cons["passed"]:
        finish("discard", "constraints", f"NaN/Inf {cons['nan_or_inf']}，越界 {cons['out_of_range']}")
    if not baseline and stats["bad_rows"] > best.get("bad_rows", stats["bad_rows"]):
        finish("discard", "constraints", "坏行增加")
    if review:
        finish("review", "lint", "; ".join(review)[:200])

    # 4. 第一道关：train 交叉验证
    if not baseline:
        ok, why = gate.stage1(cv.get("mean"), best.get("cv_accuracy"), cx["total"], best.get("complexity", 10 ** 9), hcfg)
        if not ok:
            finish("discard", "stage1", why)
        if state["dev_budget_used"] >= hcfg["dev_budget"]:
            finish("discard", "budget", "dev 评估预算已用完，请换一批 dev 标签后重置预算")

    # 5. 第二道关：dev
    dev_pairs = read_pairs(os.path.join(DEV, "pairs.tsv"))
    m = strict_pairwise(scores, dev_pairs)
    train_sites = {site_of(u) for u in train_urls}
    new_site_acc, n_new = subset_accuracy(
        m["per_pair"], lambda r: site_of(r["url_a"]) not in train_sites and site_of(r["url_b"]) not in train_sites)
    cur = {"dev_accuracy": m["accuracy"], "hard_accuracy": m["hard_accuracy"], "new_site_accuracy": new_site_acc,
           "n_new_site": n_new, "score_ties": m["n_score_tie"], "missing": m["n_missing"], "n_dev": m["n_eligible"],
           "cv_accuracy": cv.get("mean"), "complexity": cx["total"], "bad_rows": stats["bad_rows"],
           "train_accuracy": stats["train_accuracy"]}
    gpath = os.path.join(DEV, "grades.tsv")
    if os.path.exists(gpath):
        gm = grade_metrics(scores, read_grades(gpath))
        cur.update(buried_good_rate=gm.get("buried_good_rate"), top_bad_rate=gm.get("top_bad_rate"))
    bpath = os.path.join(DEV, "pairs_blind.tsv")
    if os.path.exists(bpath):
        cur["blind_accuracy"] = strict_pairwise(scores, read_pairs(bpath))["accuracy"]
    state["dev_budget_used"] += 1
    row.update(hard_acc=cur["hard_accuracy"], new_site_acc=new_site_acc, score_ties=cur["score_ties"],
               missing=cur["missing"], buried_good_rate=cur.get("buried_good_rate"), top_bad_rate=cur.get("top_bad_rate"))

    if baseline:
        state["best"] = dict(cur, run_id=run_id)
        row.update(dev_acc=m["accuracy"], dev_delta=None, threshold=None)
        finish("keep", "baseline", "基线")

    best_scores = read_scores(os.path.join(BEST, "npv_ori.tsv"))
    per_best = strict_pairwise(best_scores, dev_pairs)["per_pair"]
    comp = paired_compare(m["per_pair"], per_best)
    status, reason, eps = gate.stage2(comp["delta"], comp["se_cluster"], cur, best, cx["total"], best["complexity"], hcfg)
    row["threshold"] = eps
    if status == "keep":
        state["best"] = dict(cur, run_id=run_id)
        row.update(dev_acc=m["accuracy"], dev_delta=comp["delta"])
    else:
        row.update(dev_acc=gate.coarse(comp["delta"], eps), dev_delta=None,
                   hard_acc=None, new_site_acc=None, buried_good_rate=None, top_bad_rate=None,
                   score_ties=None, missing=None)
    finish(status, "dev", reason)


if __name__ == "__main__":
    main()
