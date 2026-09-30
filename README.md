# 🚀 Complex-IE-Bench: Evaluating LLMs on Deeply Nested JSON Extraction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Complex-IE-Bench** 是一个轻量级、高并发的大模型复杂信息抽取（Information Extraction, IE）评测与基准测试框架。

当前主流的 LLM Benchmark（如 GSM8K, MMLU）大多聚焦于单项选择或简单的数学推理。然而在真实的工业落地场景（如 FinNLP 复杂金融交易、垂直领域多意图解析）中，大模型面临的真正挑战是：**如何从高度口语化、长短句交织的非结构化文本中，稳定提取出深层嵌套的结构化 JSON 数据。**

本项目旨在填补这一空白，提供一套包含**高难度测试用例**、**异步并发评测引擎**与**多级容错评分算法**的完整 Pipeline。

## ✨ 核心特性 (Key Features)

- 📊 **深层嵌套 Schema 评测 (Deeply Nested Schema Evaluation)**
  突破传统的扁平化 Key-Value 抽取，专门评测 LLM 处理多层嵌套数组（如 `participants`, `collaterals`）、条件逻辑分支（`price_requirements`）以及多意图目标值修正（`adjusted_*`）的解耦能力。
- ⚡ **高并发异步引擎 (High-Concurrency Async Engine)**
  内置基于 `asyncio` 的请求调度器。完美支持数十个主流 LLM API（OpenAI, DeepSeek, Qwen 等），内置指数退避重试（Exponential Backoff）以应对 429 Rate Limit，并提供基于唯一键的断点续传（Checkpointing）机制，保障十万级数据请求的稳定性。
- 🧠 **多级结构化判卷算法 (Multi-Level Grading Algorithm)**
  自研针对 JSON 输出的级联评分机制，包含：
  1. 结构完整性校验（JSON 格式补全与截断修复）。
  2. Schema Key 对齐度计算。
  3. 语义级 Value 匹配（结合单位换算与符号匹配，而非死板的字符串相等）。

## 📂 仓库结构 (Repository Structure)

```text
Complex-IE-Bench/
├── dataset/
│   ├── val_seed.jsonl           # 核心测试集（脱敏口语化长文本及 Gold Standard JSON）
│   └── schema_definition.md     # 复杂嵌套 JSON 的 Schema 定义文档
├── engine/
│   ├── async_dispatcher.py      # 高并发 API 调度与断点续传模块
│   └── robust_json_parser.py    # 强力 JSON 解析与 <think> 标签过滤
├── evaluator/
│   ├── multi_level_judge.py     # 多维度判分脚本（格式、结构、数值）
│   └── statistical_metrics.py   # 计算 F1, 召回率等统计指标
├── run_benchmark.py             # 一键运行入口
└── requirements.txt
