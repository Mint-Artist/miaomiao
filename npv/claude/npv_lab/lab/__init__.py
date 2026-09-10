"""npv_lab：在冻结基线（../npv_py）之上做诊断、解释、评估与实验的代码。

导入本包会自动把 ../npv_py 加入 sys.path，之后可以直接 `from npv import ...`。
"""
import os
import sys

NPV_PY_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "npv_py"))
if NPV_PY_DIR not in sys.path:
    sys.path.insert(0, NPV_PY_DIR)

LAB_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
