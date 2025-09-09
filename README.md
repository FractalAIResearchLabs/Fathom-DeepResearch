🧮 Fathom-Search: A deep search agent
 
<div align="center">
  
[![dataset](https://img.shields.io/badge/HFData-Fathom--Search--Data-green?logo=huggingface&style=for-the-badge)](https://huggingface.co/collections/FractalAIResearch/Fathom-Search-datasets-681b42fe6f20d4b11fc51d79) 
[![space](https://img.shields.io/badge/HFSpace-Fathom--Search--4B-red?logo=huggingface&style=for-the-badge)](https://huggingface.co/spaces/FractalAIResearch/Fathom-Search-4B)

</div>

<p align="center"> <img src="./images/image.png" style="width: 100%;" id="title-icon">       </p>

---

## Overview

Large Language Models (LLMs) are getting increasingly capable but still remain bounded by static parametric knowledge and brittle retrieval pipelines. The world changes faster than any pretraining cycle can keep up and conventional RAG assumes neatly structured corpora and predictable inputs. Solving real evolving tasks requires agents that can iteratively query web, navigate noisy, heterogeneous sources, verify claims and synthesize answers under uncertainty. That is the promise of DeepSearch and the next necessary milestone on the path to reliable agents.

But scaling DeepSearch has been blocked by three hard problems: building verifiable, scalable training data that truly requires live search; stabilizing multi-turn RL with tools in the face of sparse rewards and non-stationary web environments; and overcoming lazy tool use that truncates exploration to a handful of shallow calls. 
We counter these problems with carefully designed data generation pipeline and modified optimization algorithm.

To this end, we release 

- **Fathom-Search-4B**, a 4B-parameter model trained to browse, extract, verify and reason over live web content acheiving SOTA Deep search benchmarks. Rather than memorizing facts, it learns how to find, test and trust information; sustaining deep, economical search over long horizons and noisy pages.

- **DUETQA**, a multi-agent self-play dataset of 5,000 verifiable, search-required questions that surface real-world retrieval challenges across PDFs, forums, videos, and raw HTML.

- **RAPO**, a zero-overhead modification of GRPO that stabilizes multi-turn tool use via dataset pruning, advantage scaling, and replay buffers preventing entropy collapse and enabling steady improvement under sparse rewards.

We also release **our data generation pipeline, training code, our custom web tools**  which we believe will help the community to progress further in the reasoning domain.

<!--
---

## Release Assets

- Post-Training Recipe Blog: [🤗 $499 training recipe for creating Fathom-R1-14B](https://huggingface.co/FractalAIResearch/Fathom-R1-14B)
- Final Merged Models: [🤗 Fathom-R1-14B](https://huggingface.co/FractalAIResearch/Fathom-R1-14B), [🤗 Fathom-R1-14B-RS](https://huggingface.co/FractalAIResearch/Fathom-R1-14B-RS)
- Intermediate Models:  [🤗 Fathom-R1-14B-V0.6](https://huggingface.co/FractalAIResearch/Fathom-R1-14B-V0.6), [🤗 Fathom-R1-14B-V0.4](https://huggingface.co/FractalAIResearch/Fathom-R1-14B-V0.4), [🤗 Fathom-R1-14B-V0.4-RS](https://huggingface.co/FractalAIResearch/Fathom-R1-14B-V0.4-RS)
- Fathom-R1-14B Datasets: [🤗 V0.6-Iterative-Curriculum-Learning](https://huggingface.co/datasets/FractalAIResearch/Fathom-V0.6-Iterative-Curriculum-Learning), [🤗 V0.4-SFT-Shortest-Chains](https://huggingface.co/datasets/FractalAIResearch/Fathom-V0.4-SFT-Shortest-Chains), [🤗 V0.4-RL-Compression](https://huggingface.co/datasets/FractalAIResearch/Fathom-V0.4-RL-Compression)

---
-->

## 📊 Evaluation

### Environment setup

```shell
uv venv fathom_search --python=3.10
source fathom_search/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn==2.8.2 --no-build-isolation --no-cache
playwright install
```

📊 Evaluation

We provide unified scripts under scripts/ for evaluating Fathom-Search-4B and baselines (II-Search, Jan-Nano, R1-Searcher, ZeroSearch, Search-o1).

All evaluation runs are driven by eval_search.py, with the following common arguments:

--model-path: Path to the local model checkpoint (e.g., /abs/path/to/Fathom-Search-4B)

--model-port: Port to host the model server (e.g., 1255)

--executors: Comma-separated ports for tool executors (e.g., 1211,1212)

--dataset: Dataset name (looked up under eval_benchmarks/<dataset>.jsonl) or full path to a JSONL file

--out-base: Base directory where evaluation results are stored (default: results/)

--query-llm: (Optional) Extraction model for query_url.

Can be an OpenAI model like gpt-4.1-mini (runs on CPU, needs OPENAI_API_KEY).

Or a local HuggingFace model path (must also pass --query-port to host via SGLang).

--query-port: (Optional) Port to host local query LLM if --query-llm is a path.

--main-gpus and --query-gpus: (Optional) CUDA devices to pin main model and query LLM.

🔹 Evaluate Fathom-Search-4B

Run Fathom-Search with web tools (executors) and optional query LLM.

Example 1: OpenAI extractor on CPU (main model on GPUs 0,1)
scripts/eval_fathom_search.sh \
  --model-path "model_path" \
  --model-port 1255 \
  --executors 1211,1212 \
  --dataset "dataset_name" \
  --main-gpus 0,1 \
  --query-llm gpt-4.1-mini

Example 2: Local Qwen-3B/7B as extractor (query LLM on GPUs 2,3 with TP=2)
scripts/eval_fathom_search.sh \
  --model-path "model_path" \
  --model-port 1255 \
  --executors 1211,1212 \
  --dataset "dataset_name" \
  --main-gpus 0,1 \
  --query-llm "/abs/path/to/Qwen2.5-7B-Instruct" \
  --query-port 1260 \
  --query-gpus 2,3

🔹 Evaluate II-Search-4B
scripts/eval_ii_search.sh \
  --model-path "model_path" \
  --model-port 1255 \
  --executors 1211,1212 \
  --dataset "dataset_name"

🔹 Evaluate Jan-Nano-32K
scripts/eval_jan_nano_32K.sh \
  --model-path "model_path" \
  --model-port 1255 \
  --executors 1211,1212 \
  --dataset "dataset_name"

🔹 Evaluate Jan-Nano-128K
scripts/eval_jan_nano_128K.sh \
  --model-path "model_path" \
  --model-port 1255 \
  --executors 1211,1212 \
  --dataset "dataset_name"

🔹 Evaluate R1-Searcher
scripts/eval_r1_searcher.sh \
  --model-path "model_path" \
  --model-port 1255 \
  --dataset "dataset_name"

🔹 Evaluate ZeroSearch
scripts/eval_zerosearch.sh \
  --model-path "model_path" \
  --model-port 1255 \
  --dataset "dataset_name"

🔹 Evaluate Search-o1
scripts/eval_search_o1.sh \
  --model-path "model_path" \
  --model-port 1255 \
  --dataset "dataset_name"


### Launch gradio UI.

```shell
python app.py
```

<!--

We evaluate Fathom‑R1-14B using the same metrics and sampling configuration introduced in the DeepSeek‑R1 paper, namely **pass@1** and **cons@64**. However, our evaluation is conducted under a reduced output budget of 16,384 tokens, compared to DeepSeek‑R1’s 32,768 tokens, to better reflect practical deployment constraints.

- **pass@1**: Pass@1 is computed as the average correctness over k sampled solution chains per problem (in our experiments we keep k=64).
- **cons@64**: Assesses consistency by sampling 64 reasoning chains per question and computing the majority vote accuracy.

**Evaluation Configuration**:

- Temperature: 0.6  
- top_p: 0.95  
- Number of sampled chains: 64  
- Context: 16,384 tokens  

This setup allows us to benchmark Fathom-R1-14B’s reasoning performance and stability under realistic memory and inference budgets, while maintaining compatibility with the DeepSeek‑R1 evaluation protocol.

We utilize the evaluation framework provided by the [LIMO](https://github.com/GAIR-NLP/LIMO) repository to run inference and compute metrics.
For detailed instructions and implementation details, please refer to [`eval/README.md`](./eval/readme.md).

---
-->

## Results

We evaluate **Fathom‑Search-4B** and compare with several baseline models across 9 challenging benchmarks

| Model   | SimpleQA | FRAMES  | WebWalker | Seal0 | Musique | **Avg**  | HLE  | AIME-25 | GPQA-D  | MedQA | Avg   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| <td colspan="12" style="text-align:center;">**Closed-Source Models** </td>   | 
| GPT-4o (without search)                        | 34.7                                               | 52.4                                                      | 3.2                | 7.2            | 34.0             | 26.3          | 2.3           | _71.0_   | _53.0_   | _88.2_   | 53.6          |
| o3 (without search)                            | 49.4                                               | 43.2                                                      | 14.0               | 14.0           | _48.9_      | 33.9          | _20.3_   | **88.9** | _85.4_   | **95.4** | _72.5_   |
| GPT-4o (with search)                           | _84.4_                                        | _63.7_                                               | _31.6_        | _15.3_    | 37.5             | _46.5_   | 4.3           | _71.0_   | _53.0_   | _88.2_   | 54.1          |
| o3 (with search)                               | **96.0**                                      | **86.8**                                             | **57.0**      | **49.5**  | **51.2**    | **68.1** | **27.4** | **88.9** | **85.4** | **95.4** | **74.3** |
| <td colspan="12" style="text-align:center;">**Open-Source Models** </td>   | 
| Qwen-2.5-7B                                    | 3.96                                               | 16.5                                                      | 2.1                | 1.4            | 6.2              | 6.0           | 1.2           | 10            | 33.0          | 61.2          | 24.7          |
| Qwen-2.5-7B + Search                           | 50.8                                               | 23.3                                                      | 10.1               | 3.0            | 13.6             | 20.2          | 2.4           | 10            | 33.5          | 62.0          | 25.3          |
| Qwen3-4B                                       | 3.8                                                | 14.7                                                      | 2.6                | 2.1            | 9.0              | 6.4           | 4.2           | _65.0_   | 55.1          | 71.0          | 48.8          |
| Qwen3-4B + Search                              | 67.7                                               | 27.2                                                      | 17.5               | 6.2            | 18.7             | 27.5          | 6.2           | _65.0_   | _55.9_   | 72.0          | 49.8          |
| ZeroSearch-3B                                  | 51.9                                               | 11.3                                                      | 8.7                | 7.1            | 13.8             | 18.6          | 3.4           | 10.0          | 14.6          | 51.0          | 17.3          |
| ZeroSearch-7B                                  | 75.3                                               | 30.0                                                      | 18.2               | 6.2            | 20.6             | 30.1          | 4.2           | 10.0          | 29.3          | 57.5          | 22.8          |
| R1-Searcher-7B                                 | 58.8                                               | 37.0                                                      | 1.8                | 1.4            | 19.1             | 23.6          | 2.1           | 10.0          | 33.3          | 56.5          | 25.5          |
| search-o1 (Qwen3-4B)                           | 57.5                                               | 26.8                                                      | 10.8               | 5.5            | 15.3             | 23.2          | 3.4           | 40.0          | 30.5          | 53.7          | 31.9          |
| WebSailor-3B                                   | 87.1                                               | 44.4                                                      | **52.2**      | 9.0            | 27.4             | 44.0          | 7.4           | 40.0          | 45.5          | 51.3          | 36.0          |
| Jan-Nano-32K                                   | 80.7                                               | 36.1                                                      | 25.0               | 6.2            | 21.4             | 33.9          | 5.5           | 60.0          | 37.4          | 66.0          | 42.2          |
| Jan-Nano-128K                                  | 83.2                                               | 43.4                                                      | 33.7               | 6.2            | 23.9             | 38.1          | 6.1           | 53.3          | 51.0          | 65.4          | 44.0          |
| II-Search-4B                                   | _88.2_                                        | _58.7_                                               | 40.8               | 17.1           | _31.8_      | _47.3_   | _7.4_    | 60.0          | _51.5_   | _72.1_   | 47.8          |
| **Fathom-Search-4B (Stage-1)** | 88.1                                               | 57.2                                                      | 39.0               | _19.8_    | 31.3             | 47.1          | 6.7           | 60.0          | 55.6          | **75.4** | _49.4_   |
| **Fathom-Search-4B (Stage-2)** | **90.0**                                      | **64.8**                                             | \textit{50.0}      | **22.5**  | **33.2**    | **52.1** | **9.5**  | **70.0** | **60.1** | **75.4** | **53.8** |

<!--
**Fathom‑R1-14B** demonstrates highly competitive performance across all datasets, improving over the original R1-distilled models while closely matching or surpassing other strong baselines in several settings. 
On both AIME 25 and HMMT 25, our model shows the highest pass@1 as well as cons@64 scores among all the open-source models (including the bigger R1-Distilled-32B model), with R1-670B being the only exception.

In fact, we observe that Fathom-R1-14B is superior to the first two generations of OpenAI's mini-reasoning models, including **o1-mini** and **o3-mini (low)** and it's performance closely matches that of newly released **o4-mini (low)** if provided with additional test-time compute.

---

## 🌍 Generalization Beyond Math: GPQA-Diamond

Notably, we also observe out-of-domain improvement in **GPQA-Diamond**, even though there wasn't a single instance of non-math questions in our training data. 
This indicates that our training methodology mentioned above and training on math questions facilitates generalization across diverse domains, a finding similar to what LightR1-14B & LIMO had observed.
#### ✅ GPQA Benchmark Comparison (16k)
| **Model**         | **pass@1** | **cons@64** |
|-------------------|------------|-------------|
| DeepSeek-R1-Distill-Qwen-14B          | 54.19      | 64.14       |
| LightR1‑14B                           | 56.94      | 65.15       |
| Fathom‑R1-14B-RS             | 59.13      | 66.16       |
| **Fathom‑R1-14B**            | **59.46**  | **66.16**   |

---

## Data Decontimination

Both benchmarks used (AIME 25 and HMMT 25) were released a few weeks after the release of the base model, ensuring no contamination occurred during the model's pre-training. The dataset corpora (Numina-Math 1.5 & OpenR1-Math) were released around the same time as these exams, with a cutoff date no later than 2024. Additionally, we conduct checks to verify there is no contamination in the training data.
-->


## 📜 License

This repository and all the release assets are available under the MIT License, underscoring our dedication to open and inclusive AI innovation. By freely sharing our work, we aim to democratize AI technology, empowering researchers, developers, and enthusiasts everywhere to use, adapt, and expand upon it without limitation. This open and permissive approach promotes global collaboration, accelerates innovation, and enriches the AI community as a whole.

## Acknowledgments
We would like to acknowledge the following works for enabling our project:
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- [ReCall](https://github.com/Agent-RL/ReCall/tree/main)

---

## 📖 Citation

```bibtex
@misc{fathomsearch4b2025,
  title={Fathom-Search: A deep search agent,
  author={Shreyas Singh, Pradeep Moturi and Kunal Singh},
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


 -->

 # 🧮 Fathom-Search: A deep search agent

<div align="center">

[![dataset](https://img.shields.io/badge/HFData-Fathom--Search--Data-green?logo=huggingface\&style=for-the-badge)](https://huggingface.co/collections/FractalAIResearch/Fathom-Search-datasets-681b42fe6f20d4b11fc51d79)
[![space](https://img.shields.io/badge/HFSpace-Fathom--Search--4B-red?logo=huggingface\&style=for-the-badge)](https://huggingface.co/spaces/FractalAIResearch/Fathom-Search-4B)

</div>

<p align="center"> <img src="./images/image.png" style="width: 100%;" id="title-icon">       </p>

---

## Overview

Large Language Models (LLMs) are getting increasingly capable but still remain bounded by static parametric knowledge and brittle retrieval pipelines. The world changes faster than any pretraining cycle can keep up and conventional RAG assumes neatly structured corpora and predictable inputs. Solving real evolving tasks requires agents that can iteratively query web, navigate noisy, heterogeneous sources, verify claims and synthesize answers under uncertainty. That is the promise of DeepSearch and the next necessary milestone on the path to reliable agents.

But scaling DeepSearch has been blocked by three hard problems: building verifiable, scalable training data that truly requires live search; stabilizing multi-turn RL with tools in the face of sparse rewards and non-stationary web environments; and overcoming lazy tool use that truncates exploration to a handful of shallow calls.
We counter these problems with carefully designed data generation pipeline and modified optimization algorithm.

To this end, we release

* **Fathom-Search-4B**, a 4B-parameter model trained to browse, extract, verify and reason over live web content achieving SOTA DeepSearch benchmarks. Rather than memorizing facts, it learns how to find, test and trust information; sustaining deep, economical search over long horizons and noisy pages.

* **DUETQA**, a multi-agent self-play dataset of 5,000 verifiable, search-required questions that surface real-world retrieval challenges across PDFs, forums, videos, and raw HTML.

* **RAPO**, a zero-overhead modification of GRPO that stabilizes multi-turn tool use via dataset pruning, advantage scaling, and replay buffers preventing entropy collapse and enabling steady improvement under sparse rewards.

We also release **our data generation pipeline, training code, our custom web tools** which we believe will help the community to progress further in the reasoning domain.

---

## 📊 Evaluation

> This section documents **only Fathom‑Search**: how to set up SGLang, bring up the Serper/Jina tool server, and run both batch evals and **single‑question standalone inference**.

### Environment setup

```shell
uv venv fathom_search --python=3.10
source fathom_search/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn==2.8.2 --no-build-isolation --no-cache
playwright install
```

### Model server (SGLang)

Launch the model with SGLang (change `--model-path` if you use a local path or HF repo ID):

```shell
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
        --context-length 40960
```

If you have more than 1 GPU, you can use any parallelism combinations:

```shell
        --tp 2
        --dp 2
```

### Tool server (Serper/Jina)

Set the following keys in `web_agents/host.sh`:

* **SERPER\_API\_KEY**: Obtain from serper.dev (2,500 free queries w/o CC).
* **JINA\_API\_KEY** *(optional)*: Used as a robust HTML/PDF fallback.
* **OPENAI\_API\_KEY**: Used for page/PDF summarization. (If you set it to a model URL, that endpoint will be used.)

> **Note:** To replicate our results, ensure these API keys are set.

Launch the tool server on port **8901** with 16 workers:

```bash
web_agents/host_serper.sh 8901 16
```

### Batch evaluation (optional)

The evaluator will look for `eval_benchmarks/<dataset_name>.jsonl` with columns `['id','question','answer']`.

```shell
python3 runinference.py \
    --agent recall \
    --dataset dataset_name \
    --out ./eval_results \
    --mode single \
    --workers 1 \
    --model-url "http://0.0.0.0:8902" \
    --name "fathom" \
    --limit 0,5000
```

### Standalone single‑question inference (Fathom‑Search)

1. **Start the tool server** (as above):

```bash
web_agents/host_serper.sh 8901 16
```

2. **Start the model server** (as above) on port **8902**.

3. **Ask a question with the Fathom‑Search agent**:

```bash
python ask_search.py \
  --agent fathom-search \
  --question "Who is the current RBI Governor and when did they take office?" \
  --executors http://0.0.0.0:8901 \
  --model-url http://0.0.0.0:8902 \
  --search-preset fathom
```

* You can pass multiple executors (comma‑separated) for load‑balancing, e.g. `--executors http://0.0.0.0:8901,http://0.0.0.0:8903`.
* If your agent needs tokenization hints, add `--tokenizer /path/to/Qwen3-4B`.
* Add `--no-color` to `ask_search.py` for plain (non‑Rich) output.

This prints a tidy terminal report including **Final Answer (extracted)**, a **Tool Calls** summary, and the raw **Transcript**.

## Results

We evaluate **Fathom‑Search-4B** and compare with several baseline models across 9 challenging benchmarks

| Model                                                                      | SimpleQA | FRAMES   | WebWalker | Seal0    | Musique  | **Avg**  | HLE      | AIME-25  | GPQA-D   | MedQA    | Avg      |
| -------------------------------------------------------------------------- | -------- | -------- | --------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| <td colspan="12" style="text-align:center;">**Closed-Source Models** </td> |          |          |           |          |          |          |          |          |          |          |          |
| GPT-4o (without search)                                                    | 34.7     | 52.4     | 3.2       | 7.2      | 34.0     | 26.3     | 2.3      | *71.0*   | *53.0*   | *88.2*   | 53.6     |
| o3 (without search)                                                        | 49.4     | 43.2     | 14.0      | 14.0     | *48.9*   | 33.9     | *20.3*   | **88.9** | *85.4*   | **95.4** | *72.5*   |
| GPT-4o (with search)                                                       | *84.4*   | *63.7*   | *31.6*    | *15.3*   | 37.5     | *46.5*   | 4.3      | *71.0*   | *53.0*   | *88.2*   | 54.1     |
| o3 (with search)                                                           | **96.0** | **86.8** | **57.0**  | **49.5** | **51.2** | **68.1** | **27.4** | **88.9** | **85.4** | **95.4** | **74.3** |
| <td colspan="12" style="text-align:center;">**Open-Source Models** </td>   |          |          |           |          |          |          |          |          |          |          |          |
| Qwen-2.5-7B                                                                | 3.96     | 16.5     | 2.1       | 1.4      | 6.2      | 6.0      | 1.2      | 10       | 33.0     | 61.2     | 24.7     |
| Qwen-2.5-7B + Search                                                       | 50.8     | 23.3     | 10.1      | 3.0      | 13.6     | 20.2     | 2.4      | 10       | 33.5     | 62.0     | 25.3     |
| Qwen3-4B                                                                   | 3.8      | 14.7     | 2.6       | 2.1      | 9.0      | 6.4      | 4.2      | *65.0*   | 55.1     | 71.0     | 48.8     |
| Qwen3-4B + Search                                                          | 67.7     | 27.2     | 17.5      | 6.2      | 18.7     | 27.5     | 6.2      | *65.0*   | *55.9*   | 72.0     | 49.8     |
| ZeroSearch-3B                                                              | 51.9     | 11.3     | 8.7       | 7.1      | 13.8     | 18.6     | 3.4      | 10.0     | 14.6     | 51.0     | 17.3     |
| ZeroSearch-7B                                                              | 75.3     | 30.0     | 18.2      | 6.2      | 20.6     | 30.1     | 4.2      | 10.0     | 29.3     | 57.5     | 22.8     |
| R1-Searcher-7B                                                             | 58.8     | 37.0     | 1.8       | 1.4      | 19.1     | 23.6     | 2.1      | 10.0     | 33.3     | 56.5     | 25.5     |
| search-o1 (Qwen3-4B)                                                       | 57.5     | 26.8     | 10.8      | 5.5      | 15.3     | 23.2     | 3.4      | 40.0     | 30.5     | 53.7     | 31.9     |
| WebSailor-3B                                                               | 87.1     | 44.4     | **52.2**  | 9.0      | 27.4     | 44.0     | 7.4      | 40.0     | 45.5     | 51.3     | 36.0     |
| Jan-Nano-32K                                                               | 80.7     | 36.1     | 25.0      | 6.2      | 21.4     | 33.9     | 5.5      | 60.0     | 37.4     | 66.0     | 42.2     |
| Jan-Nano-128K                                                              | 83.2     | 43.4     | 33.7      | 6.2      | 23.9     | 38.1     | 6.1      | 53.3     | 51.0     | 65.4     | 44.0     |
| II-Search-4B                                                               | *88.2*   | *58.7*   | 40.8      | 17.1     | *31.8*   | *47.3*   | *7.4*    | 60.0     | *51.5*   | *72.1*   | 47.8     |
| **Fathom-Search-4B (Stage-1)**                                             | 88.1     | 57.2     | 39.0      | *19.8*   | 31.3     | 47.1     | 6.7      | 60.0     | 55.6     | **75.4** | *49.4*   |
| **Fathom-Search-4B (Stage-2)**                                             | **90.0** | **64.8** | *50.0*    | **22.5** | **33.2** | **52.1** | **9.5**  | **70.0** | **60.1** | **75.4** | **53.8** |

---

## 📜 License

This repository and all the release assets are available under the MIT License, underscoring our dedication to open and inclusive AI innovation. By freely sharing our work, we aim to democratize AI technology, empowering researchers, developers, and enthusiasts everywhere to use, adapt, and expand upon it without limitation. This open and permissive approach promotes global collaboration, accelerates innovation, and enriches the AI community as a whole.

## Acknowledgments

We would like to acknowledge the following works for enabling our project:

* [Qwen3-4B](https://huggingface.co/Qwen/Qwen-4B)
* [ReCall](https://github.com/Agent-RL/ReCall/tree/main)

---

## 📖 Citation

```bibtex
@misc{fathomsearch4b2025,
  title={Fathom-Search: A deep search agent},
  author={Shreyas Singh and Pradeep Moturi and Kunal Singh},
  howpublished={\url{https://huggingface.co/FractalAIResearch/Fathom-Search-4B}},
  note={Hugging Face},
  year={2025}
}
```

---

## About Project Ramanujan

Fractal AI Research Lab initiated Project Ramanujan approximately one year ago, aiming to unlock intelligence and enhance AI agents by pushing the boundaries of advanced reasoning. Our key accomplishments include:

* ICLR'25 & NeurIPS'24-MATH-AI: [SBSC: Step-By-Step Coding for Improving Mathematical Olympiad Performance](https://arxiv.org/abs/2502.16666)
* Winners of HackerCupAI\@NeurIPS'24 & ICLR'25-VerifAI: [Stress Testing Based Self-Consistency For Olympiad Programming](https://openreview.net/forum?id=7SlCSjhBsq)
* CVPR'25-MULA: [TRISHUL: Towards Region Identification and Screen Hierarchy Understanding for Large VLM based GUI Agents
  ](https://arxiv.org/abs/2502.08226))
* Silver Medal in AIMO'24

