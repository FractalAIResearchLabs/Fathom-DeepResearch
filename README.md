# 🧮 Fathom-Search: Unlocking Long-Horizon DeepSearch

<div align="center">
  
[![dataset](https://img.shields.io/badge/HFData-Fathom--Search--Data-green?logo=huggingface&style=for-the-badge)](https://huggingface.co/collections/FractalAIResearch/Fathom-Search-datasets-681b42fe6f20d4b11fc51d79)
[![space](https://img.shields.io/badge/HFSpace-🚀_Chat_with_Fathom_Search-red?logo=huggingface&style=for-the-badge)](https://huggingface.co/spaces/FractalAIResearch/Fathom-Search-4B)
[![space](https://img.shields.io/badge/HFModel-Fathom--Search--4B-red?logo=huggingface&style=for-the-badge)](https://huggingface.co/FractalAIResearch/Fathom-Search-4B)

</div>

<p align="center"> <img src="./images/image.png" style="width: 100%;" id="title-icon">       </p>

---

## Introduction

Large Language Models (LLMs) have demonstrated promising results across a diverse set of tasks such as mathematical reasoning, code generation, Despite these advancements, they remain prone to factual inaccuracies/hallucinations as they rely on static internal knowledge acquired during pretraining. Real world information is continually evolving and getting updated. Given the high cost of pretraining LLMs, it is not pragmatic to rely solely on repeated pretraining to update their knowledge. A potential solution to this problem involves enabling LLMs to interface with external knowledge systems.

Retrieval-augmented generation (RAG) has become the standard framework for open-domain QA, where LLMs generate answers conditioned on retrieved context. However, these pipelines rely on structured, static corpora and predictable input formats conditions rarely met in real-world search tasks. In contrast, recent efforts have focused on instilling tool-mediated DeepSearch reasoning capabilities in LLMs that involve free-form web queries, parsing heterogeneous/noisy outputs, and synthesizing multi-step reasoning chains. The core principle underlying this approach is to concentrate training efforts on developing the model's ability to autonomously and effectively access and leverage external information sources (search engines). Rather than directly incorporating factual content into the model parameters, this method emphasizes teaching the model how to navigate, comprehend, and utilize relevant information from vast digital information landscapes for complex-information seeking tasks. This distinction makes DeepSearch fundamentally more difficult than traditional RAG and a necessary precursor to the next scaling milestone in search augmented language model capabilities: _DeepResearch Agents_

However, scaling DeepSearch capability faces three key challenges:

- the lack of high-quality, verifiable, scalable training dataset creation pipeline
- algorithmic instability in multi-turn reinforcement learning (RL) with tools
- lazy tool calling behavior which hinders scaling deep information exploration and retrieval capabilities

## ✨ Key Highlights

To this end, we introduce a post-training recipe to create state-of-the-art DeepSearch enabled reasoning model, Fathom-Search-4B. We enlist our key contributions below:

- 📚 **DUETQA**: We build a 5K example dataset created through multi-agent self-play, designed to require live web search to answer its queries.
- 🏋️ **Training Recepie**: Two-stage RL-zero training that provides coarse control over the exploration and verification strategies developed by the model.
- 🧠 **RAPO**: Zero-overhead modification of GRPO that stabilizes multi-turn RL through dataset pruning, advantage scaling, and replay buffers.
- 🛠️ **Steerable step-level reward**: We design a novel step level reward that enables fine-grained control over long-horizon tool use, which scales tool use beyond 20+ calls.
- 🏆 **Fathom-Search-4B**: SOTA 4B-parameter model trained to browse, extract, verify and reason over live web content acheiving SOTA Deep search benchmarks. Rather than memorizing facts, it learns how to find, test and trust

We also release our data generation pipeline, our custom web tools which we believe will help the community to progress further in the reasoning domain.

---

## 📊 Results

**Fathom‑Search-4B** sets a new state of the art on DeepSearch benchmarks outperforming all prior open source baselines. Our model reaches average of 52.1% on DeepSearch and 53.8% on General Reasoning benchmarks—improvements of +24.6 pp and +4.0 pp over Qwen3-4B + Search. On harder benchmarks such as FRAMES and WebWalkerQA we observe ≥100% relative gains, and > 3× on Seal0. It also exceeds prior SOTA II-Search-4B and surpasses larger Qwen2.5-7B based systems such as ZeroSearch-7B and R1-Searcher-7B

We evaluate Fathom‑Search-4B and compare with several baseline models across 9 challenging benchmarks

Model	| SimpleQA	| FRAMES	| WebWalker	| Seal0	| Musique	| Avg
--- | ---	| ---	| ---	| ---	| ---	| ---
**_Closed-Source Models_**	| ---	| ---	| ---	| ---	| ---	| ---
GPT-4o (without search)	| 34.7	| 52.4	| 3.2	| 7.2	| 34.0	| 26.3
o3 (without search)	| 49.4	| 43.2	| 14.0	| 14.0	| _48.9_	| 33.9
GPT-4o (with search)	| _84.4_	| _63.7_	| _31.6_	| _15.3_	| 37.5	| _46.5_
o3 (with search)	| **96.0**	| **86.8**	| **57.0**	| **49.5**	| **51.2**	| **68.1**
| | | | | | |
**_Open-Source Models_**	| ---	| ---	| ---	| ---	| ---	| ---
Qwen-2.5-7B	| 3.96	| 16.5	| 2.1	| 1.4	| 6.2	| 6.0
Qwen-2.5-7B + Search	| 50.8	| 23.3	| 10.1	| 3.0	| 13.6	| 20.2
Qwen3-4B	| 3.8	| 14.7	| 2.6	| 2.1	| 9.0	| 6.4
Qwen3-4B + Search	| 67.7	| 27.2	| 17.5	| 6.2	| 18.7	| 27.5
ZeroSearch-3B	| 51.9	| 11.3	| 8.7	| 7.1	| 13.8	| 18.6
ZeroSearch-7B	| 75.3	| 30.0	| 18.2	| 6.2	| 20.6	| 30.1
R1-Searcher-7B	| 58.8	| 37.0	| 1.8	| 1.4	| 19.1	| 23.6
search-o1 (Qwen3-4B)	| 57.5	| 26.8	| 10.8	| 5.5	| 15.3	| 23.2
WebSailor-3B	| 87.1	| 44.4	| **52.2**	| 9.0	| 27.4	| 44.0
Jan-Nano-32K	| 80.7	| 36.1	| 25.0	| 6.2	| 21.4	| 33.9
Jan-Nano-128K	| 83.2	| 43.4	| 33.7	| 6.2	| 23.9	| 38.1
II-Search-4B	| _88.2_	| _58.7_	| 40.8	| 17.1	| _31.8_	| _47.3_
Fathom-Search-4B (Stage-1)	| 88.1	| 57.2	| 39.0	| 19.8	| 31.3	| 47.1
**Fathom-Search-4B (Stage-2**)	| **90.0**	| **64.8**	| _50.0_	| **22.5**	| **33.2**	| **52.1**
## Results


**Generalization across domains**.
Unlike most models that drop on out-of-distribution tasks, Fathom-Search-4B generalizes well. On GPQA-D and MedQA, Fathom-Search (Stage-2) achieves 60.1% and 75.4%, surpassing WebSailor-3B and ZeroSearch-3B by +23–25 pp, despite no domain-specific finetuning.

Model	| HLE	| AIME-25	| GPQA-D	| MedQA	| Avg
--- | ---	| ---	| ---	| ---	| ---
Closed-Source Models	| ---	| ---	| ---	| ---	| ---
GPT-4o (without search)	| 2.3	| 71.0	| 53.0	| 88.2	| 53.6
o3 (without search)	| 20.3	| **88.9**	| **85.4**	| 95.4	| _72.5_
GPT-4o (with search)	| 4.3	| _71.0_	| _53.0_	| _88.2_	| 54.1
o3 (with search)	| **27.4**	| **88.9**	| **85.4**	| **95.4**	| **74.3**
Open-Source Models	| ---	| ---	| ---	| ---	| ---
Qwen-2.5-7B	| 1.2	| 10	| 33.0	| 61.2	| 24.7
Qwen-2.5-7B + Search	| 2.4	| 10	| 33.5	| 62.0	| 25.3
Qwen3-4B	| 4.2	| _65.0_	| 55.1	| 71.0	| 48.8
Qwen3-4B + Search	| 6.2	| _65.0_	| 55.9	| 72.0	| 49.8
ZeroSearch-3B	| 3.4	| 10.0	| 14.6	| 51.0	| 17.3
ZeroSearch-7B	| 4.2	| 10.0	| 29.3	| 57.5	| 22.8
R1-Searcher-7B	| 2.1	| 10.0	| 33.3	| 56.5	| 25.5
search-o1 (Qwen3-4B)	| 3.4	| 40.0	| 30.5	| 53.7	| 31.9
WebSailor-3B	| 7.4	| 40.0	| 45.5	| 51.3	| 36.0
Jan-Nano-32K	| 5.5	| 60.0	| 37.4	| 66.0	| 42.2
Jan-Nano-128K	| 6.1	| 53.3	| 51.0	| 65.4	| 44.0
II-Search-4B	| _7.4_	| 60.0	| _51.5_	| _72.1_	| 47.8
Fathom-Search-4B (Stage-1)	| 6.7	| 60.0	| 55.6	| **75.4**	| _49.4_
**Fathom-Search-4B (Stage-2)**	| **9.5**	| **70.0**	| **60.1**	| **75.4**	| 53.8

Competing with closed-source models, Fathom-Search (Stage-2) outperforms GPT-4o + Search on SimpleQA, FRAMES, WebWalkerQA, HLE, and GPQA-D. It is +18.4 pp on WebWalkerQA and +7.0 pp on both Seal0 and GPQA-D, and attains ∼2× the accuracy of GPT-4o + Search accuracy on the hard general-reasoning benchmark HLE

## 🚀 Inference

This section shows how to host the **tool web‑server** and the **SGLang model server**, then run **single‑question inference** via `inference.py`.

### 1) Environment setup

```bash
uv venv fathom_search --python=3.10
source fathom_search/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn==2.8.2 --no-build-isolation --no-cache
playwright install
```

### 2) Start the Tool Server (Serper+Jina)

Set the following in `web_agents/host.sh`:

- **SERPER_API_KEY** (get from serper.dev; ~2,500 free queries without any card) (necessary fror live web-search)
- **JINA_API_KEY** (optional) — used in the web-page extraction pipeline (recommended for replicatiion)
- **OPENAI_API_KEY** (optional) — for goal conditioned querying of web-pages using GPT-4.1-mini (recommended for replicatiion)

Launch on **port 8901** with 16 workers:

```bash
web_agents/host_serper.sh 8901 16
```

### 3) Start the Model Server (SGLang)

Change `--model-path` to your model identifier or local path (e.g., `FractalAIResearch/Fathom-Searcher` or `model_path`). Default port below is **8902**.

```bash
python -m sglang.launch_server \
  --served-model-name Qwen3-4B \
  --model-path FractalAIResearch/Fathom-Searcher \
  --enable-metrics \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8902 \
  --trust-remote-code \
  --disable-radix-cache \
  --disable-cuda-graph \
  --context-length 40960 \
  --tp 2 #optional for multi-gpu inference
```

### 4) Single‑question inference

Run **Fathom‑Search** via `inference.py`:

```bash
python inference.py \
  --agent fathom-search \
  --question "Who is the current RBI Governor and when did they take office?" \
  --executors http://0.0.0.0:8901 \
  --model-url http://0.0.0.0:8902 \
  --search-preset fathom
```

Tips:
- Use multiple executors for load‑balancing: `--executors http://0.0.0.0:8901,http://0.0.0.0:8903`.


---

## 🧮 Evaluation (Multi GPU)

This section covers **batched evaluation** using the provided scripts in `scripts/`. Use placeholders `model_path` and `dataset_name` — the evaluator will read `eval_benchmarks/<dataset_name>.jsonl` with columns `['id','question','answer']`.

### Common flags

| Flag | Required | Example | Description |
|---|:---:|---|---|
| `--model-path` | ✅ | `model_path` | Model repo ID or local path. |
| `--model-port` | ⬜ | `1255` | Port where the model server listens (if applicable). |
| `--executors` | ⬜ | `1211,1212` | Comma‑separated tool/extractor workers. |
| `--dataset` | ✅ | `dataset_name` | Looks for `eval_benchmarks/<dataset_name>.jsonl`. |
| `--out-base` | ⬜ | `./results` | Where results are written. |
| `--query-llm` | ⬜ | `gpt-4.1-mini` or `/path/to/Qwen3-4B` | Extractor/Query LLM. |
| `--query-port` | ⬜ | `1260` | Port for a locally served query LLM. |
| `--main-gpus` | ⬜ | `0,1` | CUDA devices for the main model. |
| `--query-gpus` | ⬜ | `2,3` | CUDA devices for the query LLM. |

### Evaluate Fathom‑Search (recommended starting point)

**GPT-4.1-mini query-LLM on CPU, main model on GPUs 0,1 (TP=1)**

```bash
scripts/eval_fathom_search.sh \
  --model-path model_path \
  --model-port 1255 \
  --executors 1211,1212 \
  --dataset dataset_name \
  --main-gpus 0,1 \
  --query-llm gpt-4.1-mini
```

**Local Qwen as extractor on GPUs 2,3 (TP=2); main model on GPUs 0,1 (TP=2)**

```bash
scripts/eval_fathom_search.sh \
  --model-path model_path \
  --model-port 1255 \
  --executors 1211,1212 \
  --dataset dataset_name \
  --main-gpus 0,1 \
  --query-llm /path/to/Qwen3-4B \
  --query-port 1260 \
  --query-gpus 2,3
```

### Evaluate other baselines used in the paper 

```bash
# II‑Search‑4B
scripts/eval_ii_search.sh \
  --model-path model_path \
  --model-port 1255 \
  --executors 1211,1212 \
  --dataset dataset_name

# Jan‑Nano‑32K
scripts/eval_jan_nano_32K.sh \
  --model-path model_path \
  --model-port 1255 \
  --executors 1211,1212 \
  --dataset dataset_name

# Jan‑Nano‑128K
scripts/eval_jan_nano_128K.sh \
  --model-path model_path \
  --model-port 1255 \
  --executors 1211,1212 \
  --dataset dataset_name

# R1‑Searcher (7B)
scripts/eval_r1_searcher.sh \
  --model-path model_path \
  --model-port 1255 \
  --dataset dataset_name

# ZeroSearch
scripts/eval_zerosearch.sh \
  --model-path model_path \
  --model-port 1255 \
  --dataset dataset_name

# search‑o1 (with Qwen3‑4B)
scripts/eval_search_o1.sh \
  --model-path model_path \
  --model-port 1255 \
  --dataset dataset_name
```

---

## 📜 License

This repository and all the release assets are available under the MIT License, underscoring our dedication to open and inclusive AI innovation. By freely sharing our work, we aim to democratize AI technology, empowering researchers, developers, and enthusiasts everywhere to use, adapt, and expand upon it without limitation. This open and permissive approach promotes global collaboration, accelerates innovation, and enriches the AI community as a whole.

## Acknowledgments
We would like to acknowledge the following works for enabling our project:
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen-4B)
- [ReCall](https://github.com/Agent-RL/ReCall/tree/main)

---

## 📖 Citation

```bibtex
@misc{fathomsearch2025,
  title={Fathom-Search: Unlocking Long-Horizon DeepSearch},
  author={Shreyas Singh and Pradeep Moturi and Kunal Singh},
  howpublished={\url{https://huggingface.co/FractalAIResearch/Fathom-Search-4B}},
  note={Hugging Face},
  year={2025}
}
```

---

## About Project Ramanujan

Fractal AI Research Lab initiated Project Ramanujan approximately one year ago, aiming to unlock intelligence and enhance AI agents by pushing the boundaries of advanced reasoning. Our key accomplishments include:
- ICLR'25 & NeurIPS'24-MATH-AI: [SBSC: Step-By-Step Coding for Improving Mathematical Olympiad Performance](https://arxiv.org/abs/2502.16666)
- Winners of HackerCupAI@NeurIPS'24 & ICLR'25-VerifAI: [Stress Testing Based Self-Consistency For Olympiad Programming](https://openreview.net/forum?id=7SlCSjhBsq)
- CVPR'25-MULA: [TRISHUL: Towards Region Identification and Screen Hierarchy Understanding for Large VLM based GUI Agents
](https://arxiv.org/abs/2502.08226))
- Silver Medal in AIMO'24

