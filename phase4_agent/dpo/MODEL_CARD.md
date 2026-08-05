# Model Card — DPO Fine-Tuned Web Agent (Proof of Concept)

## Base model
Qwen2.5-1.5B-Instruct (via Unsloth, 4-bit quantized, LoRA r=16)

## Training data
27 preference pairs, generated automatically from real BrowserGym/MiniWoB++
agent runs. "Chosen" = the agent's real action; "rejected" = the same action
with its element ID (`bid`) synthetically perturbed by ±5/±10.

## Training method
DPO (Direct Preference Optimization) via TRL's DPOTrainer, 3 epochs,
learning rate 5e-6, beta 0.1. Runs on a single free Kaggle T4 GPU.

## Intended use
Proof-of-concept verification that the DPO training pipeline (Unsloth + TRL,
NIM-generated preference data, Kaggle GPU) works correctly end-to-end.

## NOT intended for
Production deployment or any claim of robustly-improved agent behavior.
27 pairs is far below the scale needed for meaningful preference learning.

## Known limitations
- Training loss remained near ln(2) ≈ 0.693 (chance level) — the model did
  not demonstrably learn a stable preference at this dataset size.
- Rejected examples are synthetic (bid perturbation), not sampled from
  genuine model uncertainty or real failure cases.
- Evaluated only on MiniWoB++ tasks; no evaluation on the process-mining
  domain the original shielding graph was built from.

## Reproducing
See `phase4_agent/dpo/train_dpo.py`. Requires `configs/seeds.yaml`'s
`dpo` seed for exact reproducibility.
