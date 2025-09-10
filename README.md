# Do large language models and humans have similar behaviors in causal inference with script knowledge?

**[Hugging Face Datasets](https://huggingface.co/datasets/tonyhong/csk)** | **[arXiv e-Print](https://arxiv.org/abs/2311.07311)** | **[ACL Anthology](https://aclanthology.org/2024.starsem-1.34/)** 

> Companion repository for the *SEM 2024 paper* **“Do large language models and humans have similar behaviours in causal inference with script knowledge?”** by Xudong Hong, Margarita Ryzhova, Daniel Adrian Biondi, and Vera Demberg. 

---

## What this repo is for

This repository provides code and instructions to **evaluate language models on the CSK (Causality in Script Knowledge) corpus** and reproduce the  *model-side* analyses in the paper:

- Compute **token-level surprisal** for the critical event **B** under three conditions about its prerequisite **A**:
  - **A→B** (*control*, cause stated),
  - **¬A→B** (*failure*, cause negated),
  - **nil→B** (*target*, cause omitted).  
  CSK contains **21 stories × 3 conditions = 63 stimuli**, each split into `before B`, `B`, and `after B` text chunks. 

> The human self-paced reading experiment showed **longer reading times at B** when A was negated (¬A→B) than when A was stated (A→B). **nil→B behaved like A→B** for humans, but **LLMs generally failed** to show lower surprisal for nil→B than ¬A→B. 

---

## Environment

- Python ≥ 3.9
- PyTorch, 🤗 Transformers, 🤗 Datasets, pandas, numpy, statsmodels (for simple stats)

```bash
# Recommended
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or your CUDA wheel
pip install "transformers>=4.41" "datasets>=2.19" "pandas>=2.0" numpy "statsmodels>=0.14"
```

---

## Getting the data

Load CSK directly from Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("tonyhong/CSK")["train"]  # 63 rows total
# Fields: item, condition type (control|failure|target), before B, B, after B, ...
```

- **Label mapping used in this repo**
  - `control` → **A→B**
  - `failure` → **¬A→B**
  - `target` → **nil→B**  
  (This mapping follows the dataset card.)

---

## Evaluating models (surprisal at B)

Below is a minimal, self-contained script that computes **average per-token surprisal** of the `B` sentence given the `before B` context for any **causal LM** on 🤗 Transformers (e.g., GPT-2, OPT, GPT-NeoX, LLaMA-compatible models you have access to).

### Example runs:

```bash
python scripts/eval_surprisal.py --model gpt2
python scripts/eval_surprisal.py --model facebook/opt-1.3b
```

> **Note:** Some closed APIs do **not** expose token probabilities; the paper excluded models like ChatGPT/GPT-4 for this reason. Prefer **open models** or any API that supports logprob access. 

*Tip:* If you also want to evaluate on **TRIP** (plausible vs. implausible narratives), adapt the same surprisal procedure to its breakpoint sentence, as discussed in the paper’s comparison section. 

---

## Basic analysis

You can check whether your model mirrors the **human pattern** (higher surprisal for **¬A→B** vs **A→B**, and **nil→B ≈ A→B**):

```python
# notebooks/quick_stats.py (or run inline)
import pandas as pd
df = pd.read_csv("results_gpt2.csv")
pivot = df.pivot_table(index="condition", values="avg_surprisal", aggfunc="mean")
print(pivot.loc[["control","target","failure"]])  # control=A→B, target=nil→B, failure=¬A→B
```

For a more faithful replication, fit a (linear) mixed-effects model with **item** as a random effect on per-item averages (e.g., using `statsmodels` in Python or `lme4` in R), as done for both reading times and surprisal in the paper.

---

## Repository layout

```
.
├── scripts/
│ └── eval_surprisal.py # main evaluation script
├── notebooks/
│ └── analysis.ipynb # optional: plotting + stats
├── results/ # CSVs and summaries
├── LICENSE
└── README.md
```

---

## Key findings (from the paper)

- **Humans:**  
  - **¬A→B** (explicit causal conflict) causes **significantly longer reading times** at event B compared to **A→B**.  
  - **nil→B** behaves **similarly to A→B**, suggesting that humans can infer causality from script knowledge even without an explicit cause.

- **LLMs:**  
  - Only **some recent models** (e.g., GPT-3, Vicuna) replicate the human-like conflict effect for **¬A→B vs A→B**.
  - However, **no tested model** consistently shows lower surprisal for **nil→B** than **¬A→B**, highlighting a gap in script-based causal inference.

---

## Citation

If you use the CSK dataset or this code, please cite the original paper:

```bibtex
@inproceedings{hong-etal-2024-large,
  title     = {Do large language models and humans have similar behaviours in causal inference with script knowledge?},
  author    = {Hong, Xudong and Ryzhova, Margarita and Biondi, Daniel and Demberg, Vera},
  booktitle = {Proceedings of the 13th Joint Conference on Lexical and Computational Semantics (*SEM 2024)},
  pages     = {421--437},
  year      = {2024},
  address   = {Mexico City, Mexico},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2024.starsem-1.34/},
  doi       = {10.18653/v1/2024.starsem-1.34}
}
```

---

## Contact

Questions about the stimuli or human experiment? Contact the authors at `xLASTNAME@lst.uni-saarland.de`. 

---

### Acknowledgements

This README aggregates links and public descriptions from the ACL paper and CSK dataset card to help practitioners reproduce the model-side analyses. 

--- 
