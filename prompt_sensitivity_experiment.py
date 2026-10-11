"""
Prompt Sensitivity Experiment
==============================
How much does Qwen2.5-3B-Instruct's accuracy on a FIXED task swing when the
prompt is reformulated in ways that shouldn't matter semantically?

Axes varied (one at a time, everything else held at a shared baseline):
  (a) few-shot example order       - same 5 examples, 4 different orderings
  (b) instruction position          - before the question vs. after it
  (c) answer-prefix phrasing        - "Answer:" / "The answer is:" / "Response:"
  (d) answer option order           - A/B/C/D permuted (correct label recomputed)

Task: 120 questions from MMLU (cais/mmlu, subject = high_school_psychology).
Model: Qwen/Qwen2.5-3B-Instruct, local via transformers, fp16, no fine-tuning.
Scoring: next-token logits over the 4 letter tokens (deterministic, 1 forward
pass/question) -- not free-text generation, so results are exactly
reproducible and cheap enough for a free Colab GPU.

Run top to bottom in Google Colab with a T4 GPU runtime.
    !pip install -q transformers accelerate datasets
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MMLU_SUBJECT = "high_school_psychology"
N_EVAL = 120          # eval questions sampled from the MMLU test split
N_FEWSHOT = 5          # few-shot examples, taken from the MMLU dev split
BATCH_SIZE = 8
SEED = 0
OUT_DIR = "prompt_sensitivity_results"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

LETTERS = ["A", "B", "C", "D"]
INSTRUCTION = (
    "Answer the following multiple-choice question by choosing the single "
    "correct option (A, B, C, or D). Reply with just the letter."
)

# ============================================================
# 2. MODEL
# ============================================================
print("Loading tokenizer + model (first run downloads ~6GB, a few minutes)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"          # so the last column is always the last real token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()
device = next(model.parameters()).device
print("Model loaded on:", device)

# Token id for each letter as it appears right after a prompt ending in
# "Answer:" (no trailing space) -- the model's first generated token is
# normally " A" / " B" / etc. as a single BPE token.
LETTER_TOKEN_ID = {
    L: tokenizer.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS
}
print("Letter -> token id:", LETTER_TOKEN_ID)

# ============================================================
# 3. DATA
# ============================================================
print(f"Loading MMLU / {MMLU_SUBJECT} ...")
ds = load_dataset("cais/mmlu", MMLU_SUBJECT)

dev_pool = list(ds["dev"])            # exactly 5 examples -- our few-shot pool
test_pool = list(ds["test"])
random.shuffle(test_pool)
eval_set = test_pool[:N_EVAL]
fewshot_base = dev_pool[:N_FEWSHOT]
assert len(fewshot_base) == N_FEWSHOT, "not enough dev examples for few-shot pool"
print(f"{len(eval_set)} eval questions, {len(fewshot_base)} few-shot examples")

# ============================================================
# 4. PROMPT BUILDING
# ============================================================
def format_options(choices, option_perm):
    """choices: list[4] in original MMLU order.
    option_perm: tuple mapping NEW position -> ORIGINAL index.
    Returns the 'A. ...\\nB. ...' block in the new order."""
    perm_choices = [choices[i] for i in option_perm]
    lines = "\n".join(f"{LETTERS[i]}. {perm_choices[i]}" for i in range(4))
    return lines


def correct_letter_after_perm(answer_idx, option_perm):
    """Which new letter does the originally-correct option end up at?"""
    return LETTERS[list(option_perm).index(answer_idx)]


def build_block(question, choices_lines, instruction_position, answer_prefix, answer_letter=None):
    body = f"{question}\n{choices_lines}"
    if instruction_position == "before":
        block = f"{INSTRUCTION}\n\n{body}\n{answer_prefix}"
    else:  # "after"
        block = f"{body}\n{INSTRUCTION}\n{answer_prefix}"
    if answer_letter is not None:
        block += f" {answer_letter}"
    return block


def build_prompt(item, fewshot_items, fewshot_order, option_perm, instruction_position, answer_prefix):
    parts = []
    # Few-shot demonstrations: shown in the order under test, formatted with
    # the same instruction-position/phrasing settings under test. Their own
    # option order is left untouched (identity) so the demonstrated
    # letter->content mapping is never itself ambiguous.
    for idx in fewshot_order:
        fs = fewshot_items[idx]
        lines = format_options(fs["choices"], (0, 1, 2, 3))
        letter = LETTERS[fs["answer"]]
        parts.append(build_block(fs["question"], lines, instruction_position, answer_prefix, answer_letter=letter))

    # Target question: THIS is where option_perm is actually applied.
    lines = format_options(item["choices"], option_perm)
    target_block = build_block(item["question"], lines, instruction_position, answer_prefix)
    parts.append(target_block)

    prompt = "\n\n".join(parts)
    correct = correct_letter_after_perm(item["answer"], option_perm)
    return prompt, correct


@torch.no_grad()
def batch_predict(prompts, batch_size=BATCH_SIZE):
    preds = []
    cand_ids = torch.tensor([LETTER_TOKEN_ID[L] for L in LETTERS], device=device)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        out = model(**enc)
        last_logits = out.logits[:, -1, :]        # left-padded -> col -1 is always the real last token
        cand_logits = last_logits[:, cand_ids]     # (batch, 4)
        pred_idx = cand_logits.argmax(dim=-1).tolist()
        preds.extend(LETTERS[j] for j in pred_idx)
    return preds


# ============================================================
# 5. VARIANT DEFINITIONS
# ============================================================
random.seed(SEED)
shuffle1 = list(range(N_FEWSHOT)); random.shuffle(shuffle1)
shuffle2 = list(range(N_FEWSHOT)); random.shuffle(shuffle2)
opt_shuffle = list(range(4)); random.shuffle(opt_shuffle)

BASELINE = dict(
    fewshot_order=list(range(N_FEWSHOT)),
    instruction_position="before",
    answer_prefix="Answer:",
    option_perm=(0, 1, 2, 3),
)


def variant(**overrides):
    cfg = dict(BASELINE)
    cfg.update(overrides)
    return cfg


VARIANTS = {
    "baseline": variant(),
    # (a) few-shot example order
    "fewshot_reversed": variant(fewshot_order=list(range(N_FEWSHOT))[::-1]),
    "fewshot_shuffle_1": variant(fewshot_order=shuffle1),
    "fewshot_shuffle_2": variant(fewshot_order=shuffle2),
    # (b) instruction position
    "instruction_after": variant(instruction_position="after"),
    # (c) answer-prefix phrasing
    "phrasing_the_answer_is": variant(answer_prefix="The answer is:"),
    "phrasing_response": variant(answer_prefix="Response:"),
    # (d) answer option order
    "options_reversed": variant(option_perm=(3, 2, 1, 0)),
    "options_shuffled": variant(option_perm=tuple(opt_shuffle)),
}

AXIS_OF = {
    "baseline": None,
    "fewshot_reversed": "fewshot_order",
    "fewshot_shuffle_1": "fewshot_order",
    "fewshot_shuffle_2": "fewshot_order",
    "instruction_after": "instruction_position",
    "phrasing_the_answer_is": "phrasing",
    "phrasing_response": "phrasing",
    "options_reversed": "option_order",
    "options_shuffled": "option_order",
}

print("\nVariants to run:", list(VARIANTS.keys()))

# ============================================================
# 6. RUN ALL VARIANTS
# ============================================================
records = []
for vname, cfg in VARIANTS.items():
    print(f"Running variant: {vname}")
    prompts, corrects = [], []
    for item in eval_set:
        p, c = build_prompt(
            item, fewshot_base, cfg["fewshot_order"], cfg["option_perm"],
            cfg["instruction_position"], cfg["answer_prefix"],
        )
        prompts.append(p)
        corrects.append(c)
    preds = batch_predict(prompts)
    for qi, (item, pred, corr) in enumerate(zip(eval_set, preds, corrects)):
        records.append(dict(
            variant=vname, axis=AXIS_OF[vname], question_idx=qi,
            question=item["question"], predicted=pred, correct=corr,
            is_correct=(pred == corr),
        ))

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "raw_results.csv"), index=False)
print(f"\nSaved {len(df)} rows to {OUT_DIR}/raw_results.csv")

# ============================================================
# 7. ACCURACY TABLE + PLOT
# ============================================================
acc = df.groupby("variant")["is_correct"].mean().reindex(VARIANTS.keys())
acc_df = acc.rename("accuracy").reset_index()
acc_df.to_csv(os.path.join(OUT_DIR, "accuracy_by_variant.csv"), index=False)
print("\nAccuracy by variant:")
print(acc_df.to_string(index=False))

plt.figure(figsize=(9, 5))
plt.bar(acc_df["variant"], acc_df["accuracy"])
plt.axhline(acc.loc["baseline"], color="red", linestyle="--", linewidth=1, label="baseline")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title(f"Qwen2.5-3B-Instruct on MMLU/{MMLU_SUBJECT} (n={N_EVAL})\naccuracy across superficial prompt variants")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "accuracy_by_variant.png"), dpi=150)
plt.show()

# ============================================================
# 8. SPREAD PER AXIS -- which factor swings accuracy the most
# ============================================================
axis_rows = []
for axis in [a for a in df["axis"].dropna().unique()]:
    variants_in_axis = ["baseline"] + [v for v, a in AXIS_OF.items() if a == axis]
    sub = acc.loc[variants_in_axis]
    axis_rows.append(dict(
        axis=axis, min_acc=sub.min(), max_acc=sub.max(),
        spread=sub.max() - sub.min(), n_variants=len(sub),
    ))
axis_df = pd.DataFrame(axis_rows).sort_values("spread", ascending=False)
axis_df.to_csv(os.path.join(OUT_DIR, "spread_by_axis.csv"), index=False)
print("\nAccuracy spread by axis (max - min accuracy within that axis, baseline included):")
print(axis_df.to_string(index=False))

biggest = axis_df.iloc[0]
print(
    f"\n>>> Largest swing: '{biggest.axis}' -- {biggest.spread * 100:.1f} points "
    f"(min {biggest.min_acc:.3f}, max {biggest.max_acc:.3f})"
)

# ============================================================
# 9. QUESTION-LEVEL FLIPS vs. BASELINE
# ============================================================
base_pred = df[df.variant == "baseline"].set_index("question_idx")["predicted"]
base_correct = df[df.variant == "baseline"].set_index("question_idx")["correct"]

flip_rows = []
for vname in VARIANTS:
    if vname == "baseline":
        continue
    v_pred = df[df.variant == vname].set_index("question_idx")["predicted"]
    flipped_idx = base_pred.index[base_pred != v_pred.reindex(base_pred.index)]
    for qi in flipped_idx:
        flip_rows.append(dict(
            variant=vname, axis=AXIS_OF[vname], question_idx=qi,
            question=eval_set[qi]["question"][:150],
            correct_answer=base_correct.loc[qi],
            baseline_pred=base_pred.loc[qi],
            variant_pred=v_pred.loc[qi],
        ))

flips_df = pd.DataFrame(flip_rows)
flips_df.to_csv(os.path.join(OUT_DIR, "flips_vs_baseline.csv"), index=False)

print(f"\nTotal (variant, question) flip events vs. baseline: {len(flips_df)}")
if len(flips_df):
    print(flips_df.groupby("variant").size().sort_values(ascending=False).to_string())

    flip_counts_by_axis = flips_df.groupby("axis").size().rename("num_flips").reset_index()
    flip_counts_by_axis.to_csv(os.path.join(OUT_DIR, "flip_counts_by_axis.csv"), index=False)
    print("\nFlip count by axis:")
    print(flip_counts_by_axis.to_string(index=False))

    print("\nExample flips:")
    for _, row in flips_df.head(8).iterrows():
        print(f"- [{row.variant}] Q{row.question_idx}: {row.question}...")
        print(f"    baseline picked {row.baseline_pred}, variant picked {row.variant_pred}  (correct answer: {row.correct_answer})")

print(f"\nAll results saved under ./{OUT_DIR}/")
print("  raw_results.csv          - every (variant, question) prediction")
print("  accuracy_by_variant.csv  - accuracy per variant")
print("  accuracy_by_variant.png  - bar chart")
print("  spread_by_axis.csv       - accuracy spread per superficial-factor axis")
print("  flips_vs_baseline.csv    - every question whose predicted letter changed")
print("  flip_counts_by_axis.csv  - how many flips each axis caused")
