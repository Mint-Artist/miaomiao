#!/usr/bin/env python3
"""sandbox 静态检查。返回 (硬违规列表, 需人工审阅列表)。

硬违规：导入白名单之外的模块、读写文件、动态执行、双下划线属性、config.json 不合法、缺少 gen_features。
需审阅：新增域名字面量超过上限、名单文件行数超过上限。
"""
import ast
import glob
import os
import re
import subprocess
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_sandbox_config  # noqa: E402

ALLOWED_MODULES = {"math", "re", "urllib", "urllib.parse", "typing"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals",
                   "vars", "getattr", "setattr", "delattr", "locals", "memoryview"}
DOMAIN_RE = re.compile(r"(?<![\w.])(?:[a-z0-9-]+\.)+(?:com|cn|net|org|gov|edu|info|cc|io|me|tv|co|jp|kr|hk|tw|us|uk|de|fr|ru|biz|xyz|top|app|site)(?![\w.])", re.I)


def _domains_in_source(src: str) -> set:
    found = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.update(m.lower() for m in DOMAIN_RE.findall(node.value))
    return found


def _previous_source(rel_path: str) -> str:
    try:
        return subprocess.run(["git", "show", f"HEAD:{rel_path}"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout
    except Exception:  # noqa: BLE001  没有 git 历史时按空处理
        return ""


def lint_features(path: str, domain_limit: int) -> Tuple[List[str], List[str]]:
    hard, review = [], []
    with open(path, encoding="utf-8") as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"语法错误: {e}"], []
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "gen_features" not in names:
        hard.append("缺少 gen_features(scorer, x, site_list)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_MODULES:
                    hard.append(f"不允许导入 {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_MODULES:
                hard.append(f"不允许导入 {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            hard.append(f"不允许调用 {node.func.id}()")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__name__":
            hard.append(f"不允许访问 {node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            hard.append("不允许 global/nonlocal")
    new_domains = _domains_in_source(src) - _domains_in_source(_previous_source("sandbox/features.py"))
    if len(new_domains) > domain_limit:
        review.append(f"新增域名字面量 {len(new_domains)} 个，超过上限 {domain_limit}: {sorted(new_domains)[:10]}...")
    return hard, review


def lint_config(path: str) -> List[str]:
    try:
        cfg, fit = load_sandbox_config(path)
    except Exception as e:  # noqa: BLE001
        return [f"config.json 不合法: {e}"]
    hard = []
    for k, v in cfg.fea_weight.items():
        if not isinstance(v, (int, float)) or v != v:
            hard.append(f"fea_weight[{k}] 不是有效数字")
    return hard


def lint_lists(dir_path: str, line_limit: int) -> List[str]:
    total = 0
    for p in glob.glob(os.path.join(dir_path, "*.txt")):
        with open(p, encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return [f"lists/ 共 {total} 行，超过上限 {line_limit}"] if total > line_limit else []


def lint_sandbox(sandbox_dir: str, domain_limit: int = 20, list_lines_limit: int = 200) -> Tuple[List[str], List[str]]:
    hard, review = lint_features(os.path.join(sandbox_dir, "features.py"), domain_limit)
    hard += lint_config(os.path.join(sandbox_dir, "config.json"))
    review += lint_lists(os.path.join(sandbox_dir, "lists"), list_lines_limit)
    return hard, review


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sandbox", default=os.path.join(ROOT, "sandbox"))
    a = p.parse_args(argv)
    hard, review = lint_sandbox(a.sandbox)
    for h in hard:
        print("HARD", h)
    for r in review:
        print("REVIEW", r)
    print("ok" if not hard else "fail")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
