"""用偏序标签拟合基础分权重：P(a 比 b 好) = sigmoid(w · (f_a - f_b))，纯 Python 梯度下降。

只拟合 fea_weight 这一层线性权重；衰减、页面类别、正文长度等乘法调整保持不变，
所以拟合结果是近似最优而非全局最优。要把乘法项也纳入拟合，需要先把流水线改成单一线性形式（报告 4.3）。
"""
import math
from typing import Dict, List, Sequence, Tuple


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


def fit_pairwise_logistic(diffs: List[Sequence[float]], targets: List[float], names: Sequence[str],
                          l2: float = 1e-3, epochs: int = 500, lr: float = 0.5,
                          init: Sequence[float] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """diffs[i] = f_a - f_b，targets[i] ∈ {1, 0, 0.5}。返回 (权重, 训练指标)。"""
    k = len(names)
    w = list(init) if init is not None else [0.0] * k
    n = len(diffs)
    if n == 0:
        raise ValueError("没有可用的训练对")
    for _ in range(epochs):
        grad = [l2 * wi for wi in w]
        for d, t in zip(diffs, targets):
            p = _sigmoid(sum(wi * di for wi, di in zip(w, d)))
            err = p - t
            for j in range(k):
                grad[j] += err * d[j] / n
        w = [wi - lr * g for wi, g in zip(w, grad)]
    loss = 0.0
    correct = usable = 0
    for d, t in zip(diffs, targets):
        p = _sigmoid(sum(wi * di for wi, di in zip(w, d)))
        p = min(max(p, 1e-12), 1 - 1e-12)
        loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        if t != 0.5:
            usable += 1
            correct += int((p > 0.5) == (t == 1.0))
    return dict(zip(names, w)), {"log_loss": loss / n, "train_accuracy": correct / usable if usable else float("nan"),
                                 "n_pairs": n}


def rescale_weights(w: Dict[str, float], total_abs: float) -> Dict[str, float]:
    """把权重按绝对值之和缩放到 total_abs（基线为 80），使分数量级与基线可比。"""
    s = sum(abs(v) for v in w.values())
    return {k: v * total_abs / s for k, v in w.items()} if s > 0 else dict(w)
