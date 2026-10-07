# -*- coding: utf-8 -*-
"""
重建统一矩阵：
- 从 Phase1_Detailed_Logs.csv 取真实模型（排除已下线的旧 moonshot-* 404 记录）
- 合并 kimi_results.csv 中的 kimi-k2.6
- 输出正确矩阵 output_eval/Phase1_01_Matrix.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output_eval"

logs = pd.read_csv(OUT / "Phase1_Detailed_Logs.csv")
old = logs[~logs.model_name.str.startswith("moonshot")]
pivot = old.pivot_table(index="record_id", columns="model_name",
                        values="score", aggfunc="last")

kimi = pd.read_csv(OUT / "kimi_results.csv").set_index("record_id")
pivot["kimi-k2.6"] = kimi["kimi-k2.6"]

cols = list(pivot.mean().sort_values(ascending=False).index)
pivot = pivot[cols].reset_index()
pivot.to_csv(OUT / "Phase1_01_Matrix.csv", index=False, encoding="utf-8-sig")

print("=== 重建后各模型通过率（50 条病历）===")
for c in cols:
    print(f"{c:16s} {pivot[c].mean():.1%}")
print("\n矩阵规模:", pivot.shape)
