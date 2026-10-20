#!/usr/bin/env python3
"""探测 OpenAI 兼容模型服务是否满足 npv_ar + pi 的要求。纯标准库，内网可直接运行。

  python tools/probe_llm.py --base http://内网网关/v1 --key $KEY --model 模型名
  环境变量也可以：OPENAI_BASE_URL / OPENAI_API_KEY / NPV_AR_MODEL_ID

检查项（按顺序）：
  1. GET /models 连通
  2. 普通文本回复
  3. 非流式工具调用：返回结构化 tool_calls，参数是合法 JSON
  4. 流式工具调用：delta.tool_calls 分片能拼回完整参数，有 finish_reason，有 [DONE]   ← pi 用的是这条路径
  5. 工具结果回传：role=tool 的消息服务端能接受并继续生成
  6. 兼容性探测：store 字段、developer 角色、max_completion_tokens 是否被接受 → 给出 models.json 的 compat 建议
  7. （可选 --pi）用 pi 本体跑一句只读命令，验证 models.json 配置可用

退出码：0 全部关键项通过；1 有关键项失败；2 参数或网络错误。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

TOOL = {"type": "function", "function": {
    "name": "run_eval", "description": "运行评估",
    "parameters": {"type": "object", "properties": {"hypothesis": {"type": "string"}, "desc": {"type": "string"}},
                   "required": ["hypothesis", "desc"]}}}
ASK = "请调用 run_eval 工具，hypothesis 填 测试，desc 填 连通性检查。不要用文字回答，直接调用工具。"


class Probe:
    def __init__(self, base, key, model, timeout):
        self.base = base.rstrip("/")
        self.key = key
        self.model = model
        self.timeout = timeout
        self.results = []

    # ---------- HTTP ----------
    def _req(self, path, body=None, stream=False):
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        req.add_header("Authorization", f"Bearer {self.key}")
        req.add_header("Content-Type", "application/json")
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace"), time.time() - t0
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}", time.time() - t0
        if stream:
            return resp.status, resp, time.time() - t0
        return resp.status, resp.read().decode("utf-8", "replace"), time.time() - t0

    def chat(self, messages, tools=None, stream=False, extra=None):
        body = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"] = tools
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        if extra:
            body.update(extra)
        return self._req("/chat/completions", body, stream=stream)

    # ---------- 记录 ----------
    def record(self, name, ok, detail, critical=True):
        self.results.append((name, ok, detail, critical))
        mark = "PASS" if ok else ("FAIL" if critical else "WARN")
        print(f"[{mark}] {name}: {detail}", flush=True)
        return ok

    # ---------- 检查项 ----------
    def check_models(self):
        code, body, dt = self._req("/models")
        if code != 200:
            return self.record("1 /models 连通", False, f"HTTP {code}: {str(body)[:200]}")
        try:
            ids = [m.get("id") for m in json.loads(body).get("data", [])]
        except Exception:  # noqa: BLE001
            ids = []
        hit = self.model in ids
        return self.record("1 /models 连通", True,
                           f"{dt:.1f}s，列出 {len(ids)} 个模型" + ("，含目标模型" if hit else f"，未列出 {self.model}（有些网关不列，可忽略）"))

    def check_text(self):
        code, body, dt = self.chat([{"role": "user", "content": "只回答一个词：好的"}])
        if code != 200:
            return self.record("2 文本回复", False, f"HTTP {code}: {str(body)[:300]}")
        try:
            msg = json.loads(body)["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
        except Exception as e:  # noqa: BLE001
            return self.record("2 文本回复", False, f"响应解析失败 {e}: {str(body)[:200]}")
        return self.record("2 文本回复", bool(text), f"{dt:.1f}s，回复: {text[:60]!r}")

    def check_tool_nonstream(self):
        code, body, dt = self.chat([{"role": "user", "content": ASK}], tools=[TOOL])
        if code != 200:
            return self.record("3 非流式工具调用", False, f"HTTP {code}: {str(body)[:300]}"), None
        try:
            d = json.loads(body)
            msg = d["choices"][0]["message"]
            fin = d["choices"][0].get("finish_reason")
        except Exception as e:  # noqa: BLE001
            return self.record("3 非流式工具调用", False, f"响应解析失败 {e}"), None
        calls = msg.get("tool_calls") or []
        if not calls:
            hint = "没有 tool_calls；内容被当成文本返回。vLLM 需要 --enable-auto-tool-choice 和匹配模型的 --tool-call-parser"
            return self.record("3 非流式工具调用", False, f"{hint}。content={str(msg.get('content'))[:120]!r}"), None
        fn = calls[0].get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
            ok = fn.get("name") == "run_eval" and "hypothesis" in args and "desc" in args
            detail = f"{dt:.1f}s，name={fn.get('name')} args={args} finish_reason={fin}"
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"arguments 不是合法 JSON: {fn.get('arguments')!r} ({e})"
        if ok and fin not in ("tool_calls", "stop"):
            detail += "（finish_reason 不是 tool_calls，pi 仍可按 tool_calls 处理）"
        return self.record("3 非流式工具调用", ok, detail), calls[0]

    def check_tool_stream(self):
        code, resp, dt0 = self.chat([{"role": "user", "content": ASK}], tools=[TOOL], stream=True)
        if code != 200:
            return self.record("4 流式工具调用", False, f"HTTP {code}: {str(resp)[:300]}")
        calls = {}
        finish = None
        got_done = got_usage = False
        n_chunks = 0
        t0 = time.time()
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    got_done = True
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    return self.record("4 流式工具调用", False, f"SSE 片段不是 JSON: {payload[:120]!r}")
                n_chunks += 1
                if ev.get("usage"):
                    got_usage = True
                for ch in ev.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
        except Exception as e:  # noqa: BLE001
            return self.record("4 流式工具调用", False, f"读取流失败: {type(e).__name__}: {e}")
        dt = time.time() - t0 + dt0
        if not calls:
            return self.record("4 流式工具调用", False,
                               f"{n_chunks} 个片段里没有 delta.tool_calls（finish_reason={finish}）。流式工具解析未开启或解析器不匹配")
        slot = calls[0]
        try:
            args = json.loads(slot["args"] or "{}")
            ok = slot["name"] == "run_eval" and "hypothesis" in args and "desc" in args
        except ValueError:
            ok, args = False, slot["args"]
        notes = []
        if not slot["id"]:
            notes.append("缺 tool_call id")
        if finish is None:
            notes.append("没有 finish_reason（models.json 需设 supportsFinishReason:false）")
        if not got_done:
            notes.append("没有 [DONE]")
        if not got_usage:
            notes.append("流里没有 usage（可设 supportsUsageInStreaming:false）")
        self._stream_notes = notes
        detail = f"{dt:.1f}s，{n_chunks} 片段，name={slot['name']} args={args} finish_reason={finish}"
        if notes:
            detail += "；注意: " + "，".join(notes)
        return self.record("4 流式工具调用", ok, detail)

    def check_tool_result_roundtrip(self, call):
        if not call:
            return self.record("5 工具结果回传", False, "跳过（第 3 项未通过）")
        msgs = [{"role": "user", "content": ASK},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call.get("id") or "call_0", "name": "run_eval",
                 "content": "status=keep  #0001  dev_acc=0.7800(+0.0258)"},
                {"role": "user", "content": "用一句话复述上面工具返回的 status 和 dev_acc。"}]
        code, body, dt = self.chat(msgs, tools=[TOOL])
        if code != 200:
            return self.record("5 工具结果回传", False, f"HTTP {code}: {str(body)[:300]}（服务端的聊天模板可能不接受 role=tool）")
        try:
            msg = json.loads(body)["choices"][0]["message"]
            text = (msg.get("content") or "") + json.dumps(msg.get("tool_calls") or [], ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            return self.record("5 工具结果回传", False, f"响应解析失败 {e}")
        ok = "keep" in text or "0.78" in text
        return self.record("5 工具结果回传", ok, f"{dt:.1f}s，回复: {text[:100]!r}", critical=False)

    def check_compat(self):
        """探测 pi 对非知名地址默认会发的字段。被 4xx 拒绝的字段要在 models.json 的 compat 里关掉。"""
        base_msgs = [{"role": "user", "content": "只回答：好"}]
        probes = [
            ("store", {"store": False}, base_msgs, "supportsStore"),
            ("max_completion_tokens", {"max_completion_tokens": 16}, base_msgs, "maxTokensField"),
            ("developer 角色", {}, [{"role": "developer", "content": "你是测试助手"}] + base_msgs, "supportsDeveloperRole"),
            ("reasoning_effort", {"reasoning_effort": "low"}, base_msgs, "supportsReasoningEffort"),
        ]
        compat = {}
        for name, extra, msgs, key in probes:
            code, body, _ = self.chat(msgs, extra=extra)
            accepted = code == 200
            if key == "maxTokensField":
                compat[key] = "max_completion_tokens" if accepted else "max_tokens"
            else:
                compat[key] = bool(accepted)
            self.record(f"6 兼容性: {name}", True, "接受" if accepted else f"被拒绝 HTTP {code} → compat 里关掉", critical=False)
        for n in getattr(self, "_stream_notes", []):
            if "supportsFinishReason" in n:
                compat["supportsFinishReason"] = False
            if "supportsUsageInStreaming" in n:
                compat["supportsUsageInStreaming"] = False
        self.compat = compat

    def check_pi(self, pi_bin, provider):
        if not shutil.which(pi_bin) and not os.path.exists(pi_bin):
            return self.record("7 pi 冒烟", False, f"找不到 pi 可执行文件 {pi_bin}", critical=False)
        env = dict(os.environ, PI_OFFLINE="1", PI_TELEMETRY="0")
        cmd = [pi_bin, "-p", "--no-session", "-ne", "--no-mcp", "-nc", "--no-approve", "--model", f"{provider}/{self.model}",
               "--tools", "read", "读 program.md 的第一行，然后只回答这一行的内容。"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return self.record("7 pi 冒烟", False, "超时 180s", critical=False)
        ok = r.returncode == 0 and "autoresearch" in (r.stdout + r.stderr)
        return self.record("7 pi 冒烟", ok, f"退出码 {r.returncode}，输出: {(r.stdout or r.stderr).strip()[:120]!r}", critical=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=os.environ.get("OPENAI_BASE_URL"), help="如 http://内网网关/v1")
    p.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", "none"))
    p.add_argument("--model", default=os.environ.get("NPV_AR_MODEL_ID"))
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--pi", action="store_true", help="最后用 pi 本体跑一句只读命令（需要已配置 models.json）")
    p.add_argument("--pi-bin", default=os.environ.get("PI_BIN", "pi"))
    p.add_argument("--provider", default="intranet", help="models.json 里的 provider 名，配合 --pi")
    a = p.parse_args(argv)
    if not a.base or not a.model:
        p.error("需要 --base 和 --model（或环境变量 OPENAI_BASE_URL、NPV_AR_MODEL_ID）")
    pr = Probe(a.base, a.key, a.model, a.timeout)
    print(f"探测 {pr.base}  模型 {pr.model}\n")
    pr.check_models()
    pr.check_text()
    _, call = pr.check_tool_nonstream()
    pr.check_tool_stream()
    pr.check_tool_result_roundtrip(call)
    pr.check_compat()
    if a.pi:
        pr.check_pi(a.pi_bin, a.provider)

    crit_fail = [n for n, ok, _, c in pr.results if c and not ok]
    print("\n==== 结论 ====")
    if crit_fail:
        print("关键项未通过:", "; ".join(crit_fail))
        print("pi 依赖第 3、4 项；若无法开启服务端工具调用，改用自写驱动的文本协议。")
    else:
        print("关键项全部通过，pi 可用。")
    print("\n建议写进 ~/.pi/agent/models.json 的 provider 配置:")
    print(json.dumps({"providers": {a.provider: {
        "baseUrl": pr.base, "api": "openai-completions", "apiKey": "$INTRANET_KEY",
        "compat": pr.compat,
        "models": [{"id": pr.model, "contextWindow": 131072, "maxTokens": 8192}]}}}, ensure_ascii=False, indent=2))
    print("\ncontextWindow 与 maxTokens 请按模型实际能力改；reasoning 模型再加 \"reasoning\": true 与合适的 thinkingFormat。")
    return 1 if crit_fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(2)
