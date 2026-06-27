[English](README.md) | **简体中文**

# LAPD

本仓库包含论文 **"Alignment Imprint: Zero-Shot AI-Generated Text Detection via Provable Preference Discrepancy"** 的代码和数据。

我们提供了 LAPD 和 RAI，以及一组零样本基线方法（fast-detectgpt、lastde++、binoculars、dna-detectllm 等）的统一框架，并在 M4、DetectRL、RAID、RealDet、XSum、WritingPrompts 和 CReD 上进行了基准评测。

**论文**: [arXiv:2604.16923](https://arxiv.org/abs/2604.16923) | **代码**: [creator-xi/LAPD](https://github.com/creator-xi/LAPD)

- 4/18/2026: LAPD 预印本已在 arXiv 发布。
- 6/27/2026: 仓库提供 LAPD 实现、基线运行器、基准数据、鲁棒性评测脚本和消融实验工具。

## 简要介绍

检测 AI 生成文本是一项困难的任务，因为现代大语言模型能够生成流畅、连贯且越来越接近人类水平的文本。传统的基于似然度的检测器常常被内容本身的复杂度所干扰：一个简单的人类句子可能比一段复杂的 AI 生成文本获得更高的似然度。

LAPD 从一个不同的观察出发：面向用户的现代大语言模型都经过了有监督微调和偏好对齐。这个对齐过程会留下可衡量的分布印记。通过比较对齐模型和对应基础模型赋予文本的对数似然，我们可以捕捉到文本从对齐偏好的概率增益中获得了多少额外概率，而非仅仅来自通用语言建模。

论文将这一信号形式化为 **Alignment Imprint（对齐印记）**，并提出 **Log-likelihood Alignment Preference Discrepancy (LAPD)**，一种标准化的信息加权统计量，能够提升高熵区域的检测鲁棒性。

| 方法 | 主基准 AUROC | 跨域 AUROC | TPR @ 0.5% FPR | 时间/文本 |
| --- | ---: | ---: | ---: | ---: |
| Fast-DetectGPT | 82.26 | 95.36 | 51.79 | 0.5578s |
| DNA-DetectLLM | 86.12 | 98.36 | 59.56 | 0.7696s |
| RAI | 85.60 | 95.69 | 69.92 | 0.3506s |
| LAPD | **92.37** | **99.49** | **92.27** | 0.5792s |

主基准结果在 M4、DetectRL Multi-LLM、DetectRL Multi-Domain、RAID 和 RealDet 上取平均。跨域结果在 XSum、WritingPrompts 和 Arxiv 上取平均，生成模型为 GPT-4 Turbo、Gemini-2.0 Flash 和 Claude-3.7 Sonnet。

## 项目概述

LAPD 是一个零样本检测器，不需要在标注的 AI/人类文本上训练分类器。给定一段输入文本，它：

1. 计算基础模型下的 token 级似然度。
2. 计算对应对齐/指令微调模型下的 token 级似然度。
3. 从基础-对齐似然度差异中构造原始对齐印记 (RAI)。
4. 应用信息加权和扰动标准化得到 LAPD 分数。

实现支持：

- LAPD 和 RAI 评测。
- 标准零样本基线，包括 likelihood、entropy、LogRank、DetectGPT、Fast-DetectGPT、Lastde++、Binoculars、DNA-DetectLLM、RoBERTa 基线以及 IRM。
- 在 M4、DetectRL、RAID、RealDet、XSum、WritingPrompts、Arxiv、文本攻击数据和 CReD 上的主基准评测。
- LAPD 组件消融、文本长度、低 FPR 行为、时间效率和模型对灵敏度的消融实验。

## 环境

首先创建 Python 3.10 环境并安装依赖。

```bash
conda create -n lapd python=3.10
conda activate lapd
pip install -r requirements.txt
```

安装后验证环境：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import method.lapd; print('lapd import ok')"
```

模型通过 `method/model.py` 加载。默认缓存目录为 `./cache`。本地模型别名同样在 `method/model.py` 中定义，例如 `llama2-7b` 映射到缓存目录下的 Llama-2-7B 检查点。

推荐默认模型：

- 采样模型：`llama2-7b`
- 基础模型：`llama2-7b`
- 对齐模型：`llama2-7b-instruct`

## 运行指南

在仓库根目录运行一个小规模 LAPD 评测：

```bash
python -m method.lapd \
  --data_source main \
  --dataset xsum \
  --source_model gpt4o \
  --sampling_model llama2-7b \
  --base_model llama2-7b \
  --instruct_model llama2-7b-instruct \
  --max_samples 10 \
  --device cuda \
  --cache_dir ./cache
```

结果将保存至：

```text
./formal_exp/lapd/
```

运行完整的 LAPD 批量脚本：

```bash
./method/scripts/run_lapd.sh
```

运行基线方法：

```bash
./method/scripts/run_baselines.sh
```

可用的基线方法包括：

```text
likelihood, logrank, entropy, fast, fast_origin, detectgpt,
lastde, dna_detectllm, binoculars, binoculars_zscore,
roberta_base, roberta_large, rai, rai_double_gpu
```

实用实验脚本：

```bash
# LAPD 主评测
bash method/scripts/run_lapd.sh

# 基线评测
bash method/scripts/run_baselines.sh

# 组件消融
bash method/scripts/component_abla.sh

# 文本长度消融
bash method/scripts/ablation_text_length.sh

# 时间效率评测
bash method/scripts/time_efficiency.sh
```

## 推荐目录结构

```text
LAPD/
├── cache/                     # 模型检查点
│   ├── llama-2-7b/
│   ├── llama-2-7b-instruct/
│   ├── falcon-7b/
│   └── falcon-7b-instruct/
├── data/
│   ├── Collected data/        # XSum、WritingPrompts、Arxiv 数据
│   ├── M4/                    # M4 基准
│   ├── DetectRL/              # DetectRL 基准
│   ├── RAID/                  # RAID 基准
│   ├── RealDet/               # RealDet 基准
│   ├── Text_attack/           # 随机和真实攻击数据
│   └── CReD/                  # 中文基准数据
├── method/
│   ├── lapd.py                # LAPD 主入口
│   ├── lapd_multi_gpu.py      # 多 GPU LAPD 变体
│   ├── component_abla.py      # 组件消融入口
│   ├── args.py                # 集中式参数解析
│   ├── model.py               # 模型和分词器加载
│   ├── core/                  # 核心似然度和差异计算
│   ├── dataloader/            # 统一基准数据加载器
│   ├── baselines/             # 基线检测器实现
│   ├── scripts/               # 批量实验脚本
│   └── utils/                 # 保存、随机种子、处理等工具
├── formal_exp/                # 实验输出
├── imgs/                      # 可视化素材
├── LAPD.pdf                   # 论文草稿
├── roc_curve.pdf
├── text_length_ablation.pdf
├── time.csv
└── README.md
```

请将检测器模型下载到 `./cache` 中。`method/model.py` 中的本地模型别名对应的预期目录名为：

```text
llama2-7b          -> ./cache/llama-2-7b
llama2-7b-instruct -> ./cache/llama-2-7b-instruct
llama3.1-8b        -> ./cache/Meta-Llama-3.1-8B
llama3.1-8b-instruct -> ./cache/Meta-Llama-3.1-8B-Instruct
gpt-j-6b           -> ./cache/gpt-j-6b
gpt-j-6b-instruct  -> ./cache/gpt-j-6b-instruct
```

可通过环境变量覆盖数据和输出根目录：

```bash
export DATA_ROOT=/path/to/data
export FORMAL_EXP_ROOT=/path/to/outputs
```

## 引用

如果您觉得这项工作有帮助，请引用：

```bibtex
@article{wu2026alignment,
  title={Alignment Imprint: Zero-Shot AI-Generated Text Detection via Provable Preference Discrepancy},
  author={Wu, Junxi and Huang, Kailin and Hu, Dongjian and Chen, Bin and Wu, Hao and Xia, Shu-Tao and Zou, Changliang},
  journal={arXiv preprint arXiv:2604.16923},
  year={2026}
}
```

## 参考仓库

本项目借鉴了以下工作的思路、基线或评测规范：

- [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)
- [Binoculars](https://arxiv.org/abs/2401.12070)
- [DNA-DetectLLM](https://github.com/Xiaoweizhu57/DNA-DetectLLM)
