# pi 框架用于 npv autoresearch 的可行性评估

- 日期：2026-10-14
- 对象：pi 编程 agent（[earendil-works/pi](https://github.com/earendil-works/pi)，原 badlogic/pi-mono，MIT 协议，当前 v1.2.0）与 [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch)（v1.9.0）
- 方法：读官方文档与源码（main 分支，2026-10-13 快照），没有安装或运行。所有结论都要按第五节的清单实测确认。
- 前提：内网，只有 OpenAI 格式的 chat/completions 接口；agent 只能改 `sandbox/`，只能调用一个固定的评估命令（见 `五个问题的回答与方案修订.md` 第五节）。

---

## 一、结论

**pi 可行，但有条件，而且不要直接用 pi-autoresearch。**

| 需求 | 是否支持 | 做法与说明 |
| --- | --- | --- |
| 接内网 chat/completions 接口 | 支持 | 在 `models.json` 里配置自定义 provider，用 `compat` 关掉 OpenAI 专有字段 |
| 模型不支持原生工具调用时退回文本协议 | **不支持** | 工具调用只从接口返回的 `tool_calls` 里解析，没有文本兜底。**这是硬条件** |
| 完全离线 | 支持 | `PI_OFFLINE=1`、`PI_TELEMETRY=0`、`-ne`；也有独立二进制可用 |
| 去掉 bash | 支持 | `--tools` 是白名单，只写 `read,write,run_eval` 即可；另加 `--no-mcp` |
| 固定参数的评估工具 | 支持 | 扩展里 `registerTool` 加 `pi.exec`（不经过 shell） |
| 只许写 `sandbox/`，不许读 dev 标签 | 部分支持 | 没有内置的路径白名单，要在扩展里拦截工具调用、自己做路径归一化 |
| 每轮全新上下文 | 支持 | `-p --no-session`，每轮起一个新进程 |
| 每轮步数上限 | 部分支持 | 没有现成参数；在扩展里数轮次，超限就中止；评估工具返回后直接结束本轮；外层再加进程超时 |
| 由 `run.py` 决定保留或回退 | 支持（自己写） | 不用 pi-autoresearch 的记录工具 |
| 直接用 pi-autoresearch | 不建议 | 见第三节 |

一句话：**用 pi 当"模型调用加工具执行"的运行时，外加我们自己写的约 100 行扩展。** 扩展负责三件事：评估工具、路径守卫、轮数上限。

---

## 二、和自写驱动比

不管用不用 pi，**评估工具、路径守卫、容器隔离都要我们自己写**。pi 替代的只是自写驱动里"调用模型、解析工具调用、重试、组织对话"那 200 行左右。所以这个选择不是一锤子买卖，以后换方向的成本很低。

| | 用 pi | 自写驱动 |
| --- | --- | --- |
| 省掉的工作 | 模型调用循环、流式解析、工具执行、改文件工具；有交互界面，P2 有人值守时可以直接看它怎么想；有 JSON 事件流可以做审计 | 无 |
| 依赖 | Node ≥ 22.19，或者官方的独立二进制 | 只要 Python 和 OpenAI SDK |
| 接口不支持工具调用时 | **用不了** | 可以退回文本协议 |
| 版本风险 | 发版很快：1.0.0 到 1.2.0 只隔了 12 天，1.x 里扩展接口还有不兼容的改动。必须锁定版本，把安装包带进内网 | 无 |
| 已知问题（均未修复，或标记为不处理） | 向兼容接口发 OpenAI 专有字段导致 400/422（可用 `compat` 绕开）；GLM 返回空工具调用后会话卡死；流式解析可能静默截断工具参数；新贡献者的 issue 默认自动关闭，出问题基本只能自己修 | 出了问题自己能直接改 |
| 安全责任 | 路径守卫写在扩展里，写错了就有洞 | 工具本身就窄，出错面更小 |

**怎么选：**

- 内网接口的**工具调用稳定**，并且能把 Node 或独立二进制带进内网 → 用 pi。省事，P2 阶段还能交互观察。
- 工具调用不稳定或者不支持，或者用的是 GLM 这类有已知卡死问题的模型 → 自写驱动，用文本协议。

两条路都先过第五节的实测，再决定。

---

## 三、为什么不直接用 pi-autoresearch

pi-autoresearch 的设计和我们的方案在三处冲突：

1. **保留还是回退由 agent 自己决定。** agent 调用 `log_experiment` 时，自己填 `status`（keep 或 discard），连主指标的数值也是 agent 填的。工具层只拦一种情况：检查脚本失败时不许保留。这正是我们要避免的"agent 自己说了算"。
2. **它就是一个 shell。** `run_experiment` 底层执行 `bash -c <命令>`。从源码推断，`./.auto/measure.sh && cat 某文件` 这种写法可以绕过它对命令前缀的检查（未实测）。初始化流程还要求 agent 自己执行 `git checkout -b`。
3. **它自己执行 git。** keep 时 `git add -A` 加提交，否则 `git checkout -- .` 加 `git clean -fd`。这会和 `run.py` 的提交、回退打架。

另外，它的自动续跑是为交互模式设计的，`-p` 模式下大概率不会续跑；它对 pi 1.2.0 的兼容性也没有验证过。

**值得借的三个点**：

- `asi`：每次实验附带一段"可操作的旁注"，告诉下一轮该注意什么；
- `ideas.md` 的想法清单；
- 用"最佳改进 ÷ 中位数绝对偏差"算置信度。这个只作参考，我们的门槛仍然用 Ladder。

---

## 四、怎么配

### 4.1 接入内网接口

`~/.pi/agent/models.json`（可以用 `PI_CODING_AGENT_DIR` 换目录）。地址、密钥、模型名换成你们的；`thinkingFormat` 按模型选：

```json
{
  "providers": {
    "intranet": {
      "baseUrl": "http://内网网关/v1",
      "api": "openai-completions",
      "apiKey": "$INTRANET_KEY",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "maxTokensField": "max_tokens",
        "supportsReasoningEffort": false,
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [{ "id": "模型名", "contextWindow": 131072, "maxTokens": 8192, "reasoning": true }]
    }
  }
}
```

为什么要写 `compat`：内网地址不在 pi 的自动识别范围内，默认会按 OpenAI 的习惯发请求，包括带 `store`、用 `max_completion_tokens`、开推理时用 `developer` 角色。vLLM、SGLang 不一定认这些字段。

### 4.2 守卫扩展（示意，未实测）

文件放在 agent 写不到的位置（比如 `/opt/npv_guard/guard.ts`），用 `-e` 加载。字段名和函数签名以 pi 1.2.0 的 `docs/extensions.md` 为准；官方示例 `examples/extensions/protected-paths.ts` 就是同一个思路。

```ts
import * as path from "node:path";
import * as fs from "node:fs";
import { Type } from "typebox";

const ROOT = fs.realpathSync(process.env.NPV_AR_ROOT!);
const WRITABLE = ["sandbox", "journal.md", "ideas.md"].map(p => path.join(ROOT, p));
const UNREADABLE = ["harness/labels/dev"].map(p => path.join(ROOT, p));
const MAX_TURNS = 30;

// 归一化：去掉 @ 前缀、展开 ~、相对 cwd 解析，对已存在的部分做 realpath（防符号链接）
function norm(p: string, cwd: string): string {
  let s = p.replace(/^@/, "");
  if (s.startsWith("~")) s = path.join(process.env.HOME ?? "", s.slice(1));
  let cur = path.resolve(cwd, s), rest = "";
  while (!fs.existsSync(cur)) { rest = path.join(path.basename(cur), rest); cur = path.dirname(cur); }
  return path.join(fs.realpathSync(cur), rest);
}
const under = (p: string, d: string) => p === d || p.startsWith(d + path.sep);

export default function (pi: any) {
  let turns = 0;
  pi.on("turn_end", (_e: any, ctx: any) => { if (++turns > MAX_TURNS) ctx.abort(); });

  pi.on("tool_call", (e: any, ctx: any) => {
    const p = e.input?.path;
    if (typeof p !== "string") return;
    const abs = norm(p, ctx.cwd);
    if (e.toolName === "read" && UNREADABLE.some(d => under(abs, d)))
      return { block: true, reason: "该目录不可读" };
    if ((e.toolName === "write" || e.toolName === "edit") && !WRITABLE.some(d => under(abs, d)))
      return { block: true, reason: "只能修改 sandbox/、journal.md、ideas.md" };
  });

  pi.registerTool({
    name: "run_eval",
    description: "运行冻结的评估脚本 harness/run.py。它会自动保留或回退本次改动，并返回一行摘要。每轮只调用一次，调用后本轮结束。",
    parameters: Type.Object({ hypothesis: Type.String(), desc: Type.String() }),
    async execute(_id: string, args: any, signal: AbortSignal) {
      const r = await pi.exec("python", ["harness/run.py", "--hypothesis", args.hypothesis, "--desc", args.desc],
                              { cwd: ROOT, timeout: 300_000, signal });
      return { content: [{ type: "text", text: (r.stdout || r.stderr).slice(-2000) }], terminate: true };
    },
  });
}
```

### 4.3 启动一轮（示意）

```bash
PI_OFFLINE=1 PI_TELEMETRY=0 NPV_AR_ROOT=$PWD timeout 900 \
  pi -p --no-session -ne --no-mcp -nc --no-approve \
     --tools read,write,run_eval \
     -e /opt/npv_guard/guard.ts \
     --append-system-prompt program.md \
     "开始一轮实验：先读 results.tsv 最后 20 行和 ideas.md，然后提出一个假设。"
```

- 每个参数的作用：
  - `-ne`：关掉内置扩展；
  - `-nc`：不加载 `AGENTS.md`；
  - `--no-approve`：不信任项目目录里的扩展和设置，避免 agent 往 `.pi/` 里塞东西改变下一轮的行为。
- 选择模型的参数写法以 `docs/cli.md` 为准。
- 工具只给 `read,write`，不给 `edit`：`edit` 要求精确匹配原文，能力一般的模型容易匹配失败；整文件重写更稳。
- 外层用一个 shell 循环反复启动，并检查 `STOP` 文件和 `results.tsv` 的行数。
- 打分步骤仍然放进容器、不挂载标签目录（修订文档 5.4），这和用不用 pi 无关。

---

## 五、一小时实测清单（到内网先做）

1. 用独立二进制在断网机器上跑 4.3 的命令，用 `tcpdump` 确认除了模型接口以外没有任何出网连接。
2. 看 vLLM 的请求日志：`store`、`developer`、`max_tokens`、推理相关字段是否符合预期；流式返回的工具调用能否正确解析。
3. 故意让模型去读 `harness/labels/dev/` 下的文件、写 `../x`、通过符号链接写、用 `@` 和 `~` 开头的路径、调用 bash，确认全部被拦下。
4. 确认 `run_eval` 返回后进程正常退出，记下退出码。
5. 连续跑 20 轮，统计工具调用失败率，观察有没有空工具调用导致的卡死。
6. 测一下系统提示加 `program.md` 的总长度，看模型能否每轮只调用一次 `run_eval`。

第 2 项和第 5 项不过关，就改用自写驱动。
