/**
 * npv_ar 守卫扩展（pi v1.3.0）。三件事：
 *  1. 拦截工具调用：只许写 sandbox/、journal.md、ideas.md；只许读仓库内文件，且不许读 dev 标签与 holdout；禁 bash。
 *  2. 注册 run_eval（固定调用 harness/run.py，不经过 shell）与 show_errors（train 错例）。
 *  3. 轮数上限：超过 NPV_AR_MAX_TURNS 就中止本轮。
 * 放在 agent 写不到的位置，用 `pi -e /abs/path/guard.ts` 加载。
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const ROOT = fs.realpathSync(process.env.NPV_AR_ROOT ?? process.cwd());
const PYTHON = process.env.NPV_AR_PYTHON ?? "python3";
const MAX_TURNS = Number(process.env.NPV_AR_MAX_TURNS ?? "30");
const EVAL_TIMEOUT_MS = Number(process.env.NPV_AR_EVAL_TIMEOUT_MS ?? "600000");
const PRIVATE_DIR = path.resolve(ROOT, process.env.NPV_AR_PRIVATE ?? "../npv_ar_private");

const WRITABLE = ["sandbox", "journal.md", "ideas.md"].map((p) => path.join(ROOT, p));
const UNREADABLE = [path.join(ROOT, "harness", "labels", "dev"), PRIVATE_DIR, path.join(ROOT, ".git"), path.join(ROOT, ".pi")];
const BLOCKED_TOOLS = new Set(["bash", "powershell"]);

/** 路径归一化：去掉 @ 前缀、展开 ~、相对 cwd 解析、对已存在的前缀做 realpath（防符号链接逃逸）。 */
function normalize(p: string, cwd: string): string {
  let s = p.trim().replace(/^@/, "");
  if (s === "~" || s.startsWith("~/")) s = path.join(process.env.HOME ?? "", s.slice(1));
  let cur = path.resolve(cwd, s);
  let rest = "";
  while (!fs.existsSync(cur)) {
    rest = rest ? path.join(path.basename(cur), rest) : path.basename(cur);
    const parent = path.dirname(cur);
    if (parent === cur) break;
    cur = parent;
  }
  return rest ? path.join(fs.realpathSync(cur), rest) : fs.realpathSync(cur);
}
const under = (p: string, dir: string) => p === dir || p.startsWith(dir + path.sep);

function pathsOf(input: Record<string, unknown>): string[] {
  const out: string[] = [];
  for (const k of ["path", "file", "file_path", "dir", "directory"]) {
    const v = input?.[k];
    if (typeof v === "string") out.push(v);
  }
  return out;
}

const runEval = defineTool({
  name: "run_eval",
  label: "run_eval",
  description:
    "运行冻结的评估脚本 harness/run.py：静态检查 -> 打分 -> 两道关 -> 自动保留或回退 -> 记录到 results.tsv。" +
    "返回一行摘要。每轮只能调用一次，调用后本轮结束。",
  promptSnippet: "run_eval: 评估 sandbox 的当前改动并自动保留或回退",
  parameters: Type.Object({
    hypothesis: Type.String({ minLength: 1, description: "本次改动的假设与预期影响" }),
    desc: Type.String({ minLength: 1, description: "一句话描述改动" }),
  }),
  async execute(_id, params, signal, _onUpdate, _ctx) {
    const r = await pi_exec(PYTHON, [path.join(ROOT, "harness", "run.py"), "--hypothesis", params.hypothesis, "--desc", params.desc], signal);
    const text = [r.stdout.trim(), r.stderr.trim()].filter(Boolean).join("\n").slice(-3000) || `(无输出，退出码 ${r.code})`;
    return { content: [{ type: "text", text }], details: { code: r.code, killed: r.killed }, terminate: true };
  },
});

const showErrors = defineTool({
  name: "show_errors",
  label: "show_errors",
  description: "列出当前最好版本在 train 偏序对上判错的对（按错得最有把握排序），以及前几条的特征差。只看 train。",
  promptSnippet: "show_errors: 查看 train 错例找灵感",
  parameters: Type.Object({ top: Type.Optional(Type.Integer({ minimum: 1, maximum: 100, description: "显示条数，默认 20" })) }),
  async execute(_id, params, signal) {
    const r = await pi_exec(PYTHON, [path.join(ROOT, "harness", "errors.py"), "--top", String(params.top ?? 20)], signal);
    const text = [r.stdout.trim(), r.stderr.trim()].filter(Boolean).join("\n").slice(-6000) || `(无输出，退出码 ${r.code})`;
    return { content: [{ type: "text", text }], details: { code: r.code } };
  },
});

let pi_exec: (cmd: string, args: string[], signal?: AbortSignal) => Promise<{ stdout: string; stderr: string; code: number; killed: boolean }>;

export default function (pi: ExtensionAPI) {
  pi_exec = (cmd, args, signal) => pi.exec(cmd, args, { cwd: ROOT, timeout: EVAL_TIMEOUT_MS, signal });
  let turns = 0;
  let evals = 0;

  pi.on("tool_call", async (event, ctx) => {
    const name = event.toolName;
    if (BLOCKED_TOOLS.has(name)) return { block: true, reason: `${name} 在本任务中不可用` };
    if (name === "run_eval") {
      if (evals >= 1) return { block: true, reason: "每轮只能调用一次 run_eval，本轮已结束", terminate: true };
      evals += 1;
      return undefined;
    }
    const input = event.input as Record<string, unknown>;
    for (const raw of pathsOf(input)) {
      let abs: string;
      try {
        abs = normalize(raw, ctx.cwd);
      } catch (e) {
        return { block: true, reason: `路径无法解析: ${raw}` };
      }
      if (name === "write" || name === "edit") {
        if (!WRITABLE.some((d) => under(abs, d))) return { block: true, reason: `只能修改 sandbox/、journal.md、ideas.md，不能写 ${raw}` };
      } else {
        if (!under(abs, ROOT) || UNREADABLE.some((d) => under(abs, d))) return { block: true, reason: `不允许读取 ${raw}` };
      }
    }
    return undefined;
  });

  pi.on("turn_end", async (_event, ctx) => {
    turns += 1;
    if (turns >= MAX_TURNS) {
      if (ctx.hasUI) ctx.ui.notify(`达到轮数上限 ${MAX_TURNS}，中止本轮`, "warning");
      ctx.abort();
    }
    return undefined;
  });

  pi.registerTool(runEval);
  pi.registerTool(showErrors);
}
