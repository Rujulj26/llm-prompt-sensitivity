# Prompt Sensitivity Experiment — Qwen2.5-3B-Instruct

How much does a fixed model's accuracy on a fixed task swing when the prompt
is reformulated in ways that shouldn't matter semantically?

**Answer, from this run:** accuracy moves 2.5–3.3 points depending on which
superficial factor you change, and a naive "did the predicted letter
change" metric overstates that instability by roughly 10x for one of the
four factors. See [Results](#results) below.

## Setup

- **Task:** 120 questions from MMLU (`cais/mmlu`, subject
  `high_school_psychology`), a standard 4-way multiple-choice benchmark.
- **Model:** `Qwen/Qwen2.5-3B-Instruct`, loaded locally via `transformers`
  in fp16. No fine-tuning, no API keys, runs on a free Colab T4 GPU.
- **Scoring:** next-token logits over the 4 letter tokens (A/B/C/D) — one
  deterministic forward pass per question, not free-text generation. Since
  this is deterministic, each variant was scored once over the full
  120-question set rather than repeated.
- **Variants:** 1 shared baseline (few-shot order 1–5, instruction before
  the question, `"Answer:"`, options in dataset order) plus 8 single-factor
  changes, one axis at a time:
  - **(a) few-shot order:** reversed, and two random shuffles of the same 5 examples
  - **(b) instruction position:** after the question instead of before
  - **(c) phrasing:** `"The answer is:"`, `"Response:"`
  - **(d) answer option order:** reversed (D,C,B,A), and one random permutation — with the correct letter recomputed each time, so the *task* never changes, only its surface form

## Files

| File | Contents |
|---|---|
| `prompt_sensitivity_experiment.py` | full pipeline as one script |
| `prompt_sensitivity_experiment.ipynb` | same pipeline, cell-by-cell, ready for Colab |
| `results/raw_results.csv` | every (variant, question) prediction, correct answer, correctness |
| `results/accuracy_by_variant.csv` | accuracy for each of the 9 variants |
| `results/accuracy_by_variant.png` | bar chart of accuracy per variant vs. baseline |
| `results/spread_by_axis.csv` | max − min accuracy within each axis |
| `results/flips_vs_baseline.csv` | every question whose predicted **letter** changed, per variant |
| `results/flip_counts_by_axis.csv` | raw letter-flip counts per axis — **see caveat below, this overstates option-order instability** |
| `results/genuine_content_flips.csv` | option-order flips filtered down to cases where the model actually picked a *different underlying option*, not just a relabeled one |

## Results

Run on the 120-question set above, baseline accuracy 85.8%:

| Variant | Accuracy | Δ vs baseline |
|---|---|---|
| baseline | 85.8% | — |
| fewshot_reversed | 85.0% | −0.8 |
| fewshot_shuffle_1 | 86.7% | +0.8 |
| fewshot_shuffle_2 | 85.0% | −0.8 |
| instruction_after | 83.3% | **−2.5** |
| phrasing_"the answer is:" | 86.7% | +0.8 |
| phrasing_"Response:" | 87.5% | +1.7 |
| options_reversed | 85.8% | 0.0 |
| options_shuffled | 82.5% | **−3.3** |

**Spread by axis** (max − min accuracy, baseline included):
option_order (3.3pp) > instruction_position (2.5pp) > fewshot_order ≈
phrasing (1.7pp each).

### Caveat: raw flip counts overstate option-order instability by ~10x

`flip_counts_by_axis.csv` reports **200 letter-level flips** for
`option_order` against 3–5 for every other axis, which looks like option
order causes 40–60x more instability than anything else. It doesn't. When
you reverse or shuffle the options, **the correct letter itself moves**, so
a model that correctly tracks content is *expected* to change its
predicted letter even when it's still picking the right answer.

Re-deriving which underlying option (not which letter) the model actually
picked in each case:

| Variant | Letters changed | Genuinely different option chosen |
|---|---|---|
| options_reversed | 116/120 | **9/120** |
| options_shuffled | 84/120 | **11/120** |

Predicted-letter distributions also shift in line with where the correct
answer moved rather than piling onto a fixed favorite letter — no evidence
of positional bias (e.g., always guessing "A"). The genuinely meaningful
option-order sensitivity is closer to **8–9% of questions**, not the ~85%
the raw flip count implies.

### What actually drove the two option-order variants apart

Both variants have similar genuine flip *counts* (9 vs. 11), but very
different net effects, because the flips point in different directions:

| | correct→wrong | wrong→correct | wrong→wrong | net accuracy Δ |
|---|---|---|---|---|
| options_reversed | 3 | 3 | 3 | 0.0pp (gains cancel losses) |
| options_shuffled | 7 | 3 | 1 | **−3.3pp** |

`instruction_after` is the second-biggest driver and the most
**one-directional**: all 3 flips went correct→wrong, none the other way —
putting the instruction after the question consistently costs accuracy
rather than just adding noise.

### Specific questions that flipped

- **Q58** ("An American teenager's prototype of a chair...") flips B→A
  (correct→wrong) under `fewshot_reversed`, `fewshot_shuffle_2`, **and**
  `instruction_after` — fragile across two independent axes.
- **Q18** ("...example of an implicit memory?") flips under both phrasing
  variants (A→D, wrong→wrong) **and** under both option-order variants —
  the model never seems confident here regardless of format.
- **Q70** (the Lupe/dating-Craig ambivalence question) flips under
  `phrasing_response` (wrong→correct) **and** under both option-order
  variants.
- **Q60** (electrochemical/neural transmission) flips under
  `fewshot_shuffle_1` (wrong→correct) and under `options_reversed`.

Across all 9 variants, **17/120 questions (14%) flip at least once**, and 4
of those (Q18, Q58, Q60, Q70) flip under **two independent axes** —
consistent with instability concentrating in a small set of low-confidence
or borderline items rather than being spread evenly across the set.

### Bottom line

On this 120-question MMLU slice, Qwen2.5-3B-Instruct's accuracy swings
2.5–3.3 points across superficial reformulations, with **instruction
position** and **genuine (content-level) option-order changes** as the real
drivers — few-shot order and phrasing barely move the needle. The more
useful finding, though, is that instability isn't uniform: it clusters in
a small set of borderline questions that flip under multiple unrelated
perturbations. A natural follow-up: check whether those items also have
the smallest logit margin between the top two candidate letters, which
would suggest confidence margin predicts prompt sensitivity.

## How to reproduce

1. Open `prompt_sensitivity_experiment.ipynb` in Google Colab.
2. `Runtime > Change runtime type > T4 GPU` (free tier).
3. Run all cells top to bottom (~10–20 minutes, mostly the one-time ~6GB
   model download).
4. Results land in `prompt_sensitivity_results/`, matching the files
   listed above.

## Limitations

- Single subject, single model, single run per variant — a small, targeted
  probe of prompt sensitivity, not a comprehensive benchmark. Swap
  `MMLU_SUBJECT` or `N_EVAL` at the top of the script for a different or
  larger slice.
- Few-shot examples keep their own option order fixed across all variants,
  so the option-order axis isolates the *target question's* formatting
  only, not the demonstrations'.
- Results above are from one fixed random seed (`SEED = 0`); the specific
  few-shot shuffles and option permutation are reproducible from that seed
  but arbitrary in choice.
