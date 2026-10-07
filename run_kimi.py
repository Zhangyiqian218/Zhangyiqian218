# -*- coding: utf-8 -*-
"""
独立补跑脚本：仅用 kimi-k2.6（Moonshot）对 50 条病历做 L0 结构化抽取。
复用 phase1.py 的 StructFlowExtractor 与 SYSTEM_PROMPT，输入构造与主流程一致。
- 该 Kimi 组织并发上限为 1，必须串行
- 每条立即落盘 + 断点续跑，中断不丢数据
结果输出到 output_eval/kimi_results.csv
"""
import os
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from phase1 import StructFlowExtractor, SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA = str(ROOT / "data" / "medical_rounds_50.xlsx")
OUT_CSV = str(ROOT / "output_eval" / "kimi_results.csv")
MODEL = "kimi-k2.6"
BASE_URL = "https://api.moonshot.cn/v1"


def build_input(row):
    context_parts = []
    for key in ["记录日期时间", "年龄", "性别"]:
        if key in row and pd.notna(row[key]):
            context_parts.append(f"{key}:{row[key]}")
    user_input = f"[辅助上下文]: {' '.join(context_parts)}\n" if context_parts else ""
    user_input += f"[口述全文]:\n{row.get('口述全文', '')}"
    return user_input


def main():
    df = pd.read_excel(DATA)
    for c in ["记录编号", "口述全文"]:
        if c not in df.columns:
            raise KeyError(f"数据源缺少必要列: {c}")
    df["记录编号"] = df["记录编号"].astype(str)
    records = df.to_dict("records")

    # 断点续跑：读已完成记录
    done = set()
    if os.path.exists(OUT_CSV):
        try:
            done = set(pd.read_csv(OUT_CSV)["record_id"].astype(str))
        except Exception:
            done = set()
    write_header = not os.path.exists(OUT_CSV)

    extractor = StructFlowExtractor(os.getenv("MOONSHOT_API_KEY"), BASE_URL, MODEL)
    pending = [r for r in records if r["记录编号"] not in done]
    print(f"共 {len(records)} 条，已完成 {len(done)}，待跑 {len(pending)}", flush=True)

    for i, row in enumerate(pending, 1):
        rid = row["记录编号"]
        t0 = time.time()
        try:
            raw, cleaned, parsed, err = extractor.evaluate_zero_shot(SYSTEM_PROMPT, build_input(row))
            rec = {"record_id": rid, MODEL: 1 if parsed else 0,
                   "latency_sec": round(time.time() - t0, 1),
                   "error_type": err or ""}
        except Exception as e:
            rec = {"record_id": rid, MODEL: 0,
                   "latency_sec": round(time.time() - t0, 1),
                   "error_type": f"脚本异常: {type(e).__name__} {str(e)[:120]}"}

        pd.DataFrame([rec]).to_csv(
            OUT_CSV, mode="a", header=write_header,
            index=False, encoding="utf-8-sig")
        write_header = False
        print(f"[{i}/{len(pending)}] {rid} -> {rec[MODEL]} ({rec['latency_sec']}s) {rec['error_type']}", flush=True)

    res = pd.read_csv(OUT_CSV)
    print(f"\n=== {MODEL} 通过率: {res[MODEL].mean():.1%}  均延迟: {res.latency_sec.mean():.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
