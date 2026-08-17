<div align="center">
  <h1>MathForm: Scaling Mathematical Autoformalization with Knowledge Retrieval and Verification-Guided Refinement</h1>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2608.14221"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="论文"></a>
  <a href="https://huggingface.co/datasets/openbmb/FormalVerse"><img src="https://img.shields.io/badge/🤗%20Dataset-FormalVerse-yellow.svg" alt="FormalVerse 数据集"></a>
  <a href="https://huggingface.co/openbmb/MathForm-8B"><img src="https://img.shields.io/badge/🤗%20Model-MathForm--8B-informational.svg" alt="MathForm-8B 模型"></a>
</p>

<p align="center">
  <img src="./assets/data-pipeline.png" width="800" alt="MathForm 数据构造与训练流程">
  <br>
  <em>图 1：MathForm 数据构造与训练流程概览。系统结合 Mathlib 知识检索、编译与语义验证以及迭代式优化，生成可靠的形式化数据，随后进行轨迹重构并训练 MathForm-8B。</em>
</p>

## 📖 简介

我们提出 **MathForm**，这是一个将 Mathlib 知识检索与验证引导的迭代式优化相结合的自动形式化框架。MathForm 在生成前检索相关定义和已有形式化结果，并利用编译器诊断与语义一致性反馈优化生成的 Lean 定理陈述。

基于该框架，我们构建了覆盖多种数学领域和数据来源的、经过验证的 Lean 4 数据集 [**FormalVerse**](https://huggingface.co/datasets/openbmb/FormalVerse)。我们还通过监督微调以及基于 Lean 编译和语义一致性反馈的强化学习训练了 [**MathForm-8B**](https://huggingface.co/openbmb/MathForm-8B)。本仓库提供数据构造流程和自动形式化评测代码。

<p align="center">
  <img src="./assets/pass8_avg.png" width="450" alt="Pass@8 平均结果">
  <br>
  <em>图 2：专用自动形式化模型在 FormalMATH-Lite、ProverBench、CombiBench、FATE-M、FATE-H 和 FATE-X 六个基准上的宏平均 Pass@8（%）。在该类别中，尽管模型规模更小，MathForm-8B 仍取得了最强的总体性能。</em>
</p>

## 新闻

- `[2026.08.14]`：MathForm 论文、代码和数据正式发布。

## 📁 仓库结构

```text
src/                         数据构造流程
evaluation/                  评测流程及基准文件
kimina-lean-server/          Lean 编译服务器源码
assets/                      README 中使用的图片
requirements.txt             Python 依赖
```

## 🛠️ 快速开始

### 安装

1. 克隆仓库：

```bash
git clone https://github.com/OpenBMB/MathForm.git
cd MathForm
```

2. 安装 Python 依赖：

```bash
pip install -r requirements.txt
```

实验使用 Lean 4.21.0。

### 启动 Kimina Lean Server

评测和数据构造流程需要运行 Kimina Lean Server 进行编译检查。

```bash
cd kimina-lean-server
cp .env.template .env
bash setup.sh
pip install -r requirements.txt
pip install .
prisma generate
python -m server
```

默认服务地址为 `http://localhost:8000`。

### 运行数据构造

输入为 JSONL 文件，每条记录应在 `statement` 或 `informal_statement` 等字段中包含自然语言数学陈述。

```bash
cd src
API_URL=https://api.example.com/v1/chat/completions \
API_KEY="$API_KEY" \
BASE_MODEL_NAME=[GENERATION_MODEL] \
JUDGE_MODEL_NAME=[JUDGE_MODEL] \
LEAN_SERVER_URL=http://localhost:8000 \
bash run.sh path/to/input.jsonl output/run
```

使用 `run_leanexp_server.sh` 启动 Lean Explore，并添加：

```bash
LEAN_EXPLORE_URL=http://localhost:9000
```

生成文件包括 `success.jsonl`、`failed.jsonl` 和 `pipeline.log`。成功样本可以使用以下命令进行规范化和过滤：

```bash
python postprocess.py normalize \
  --input output/run/success.jsonl \
  --output output/run/normalized.jsonl
python postprocess.py filter \
  --input output/run/normalized.jsonl \
  --output output/run/filtered.jsonl
```

### 运行评测

默认评测使用 `evaluation/benchmarks/` 下的基准文件。

```bash
cd evaluation
EVAL_API_BASE_URL=https://api.example.com/v1 \
EVAL_API_MODEL=[EVALUATION_MODEL] \
JUDGE_API_BASE_URL=https://api.example.com/v1 \
JUDGE_API_MODEL=[JUDGE_MODEL] \
API_KEY="$API_KEY" \
bash run.sh
```

结果写入 `evaluation/output/`：

```text
predictions.jsonl       生成的 Lean 候选
results.jsonl           编译和评测结果
results.compile.jsonl   编译缓存
results.summary.json    Pass@k 汇总结果
```

如需评测其他基准或修改采样数量，请在运行 `run.sh` 前设置 `DATASET_PATHS` 或 `NUM_SAMPLES`。

## 🔎 引用

如果本仓库对你的研究有所帮助，请引用我们的论文：

```bibtex
@misc{pu2026mathformscalingmathematicalautoformalization,
      title={MathForm: Scaling Mathematical Autoformalization with Knowledge Retrieval and Verification-Guided Refinement},
      author={Lushi Pu and Weiming Zhang and Xinheng Xie and Zixuan Fu and Bingxiang He and Hengyu Zhao and Hongya Lyu and Xin Li and Jie Zhou and Yudong Wang},
      year={2026},
      eprint={2608.14221},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.14221},
}
```

## 🤝 致谢

本仓库基于以下开源项目构建：

- [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server)，用于 Lean 编译检查。
- [Lean Explore](https://github.com/justincasher/lean-explore)，用于检索。

评测使用以下基准：

- [FormalMATH-Lite](https://huggingface.co/datasets/SphereLab/FormalMATH-Lite)
- [ProverBench](https://huggingface.co/datasets/deepseek-ai/DeepSeek-ProverBench)
- [CombiBench](https://huggingface.co/datasets/AI-MO/CombiBench)
- [FATE-M](https://github.com/frenzymath/FATE-M/tree/main)
- [FATE-H](https://github.com/frenzymath/FATE-H/tree/main)
- [FATE-X](https://github.com/frenzymath/FATE-X/tree/main)

FormalVerse 的部分自然语言题目来自以下公开数据集：

- [Lean-Workbook](https://huggingface.co/datasets/pkuAI4M/LeanWorkbook)
- [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)
- [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
- [DeepTheorem](https://huggingface.co/datasets/Jiahao004/DeepTheorem)
- [AceReason-Math](https://huggingface.co/datasets/nvidia/AceReason-Math)
- [OpenR1-Math](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [Principia-Collection](https://huggingface.co/datasets/facebook/principia-collection)

感谢这些项目的作者和贡献者。

## 📜 许可证

本项目采用 Apache License 2.0。仓库中包含的第三方组件保留其原始许可证和署名信息。
