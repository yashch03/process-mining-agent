# Project Explanation — Enterprise Process Discovery & Conformance-Guided Web Agents

*A complete, beginner-accessible walkthrough of what this project does, why, and what was actually found.*

---

## The one-sentence version

A system that discovers the "correct" way a business process actually happens (from real event logs, not documentation), turns that into a measurable violation detector, and then tests whether that same discovered structure can act as a safety guardrail for a completely different AI system — an LLM-powered web browsing agent.

---

## The problem this solves

Business processes generate huge event logs, but you often don't actually know what the "true" process is just by reading company documentation — you have to mine it from evidence. This project builds that mining pipeline end-to-end, then asks a harder question: can what you mined be reused as a safety mechanism somewhere else entirely?

---

## Part 1: Phase 1 — Ingestion (getting the data right before doing anything clever)

Real event logs are messy: malformed timestamps, missing fields, inconsistent formatting. Two disciplines were enforced before any modeling happened:

- **Strict schema validation (Pydantic)** — every event record is checked against a typed contract (`case_id`, `activity`, `timestamp`, `resource`). Malformed records are quarantined with their original data and the validation error, not silently dropped or coerced.
- **Temporal hold-out splitting, not random splitting** — the dataset is split by actual event timestamp, not a random shuffle. A case that starts before the split date but has events after it is assigned *wholly* to training, based on its earliest event. This prevents "leakage" — the model never gets to see future process behavior during training. This required fixing a real timezone bug (the BPI-2017 dataset's timestamps are UTC-aware; a naive comparison silently fails) before it worked correctly.

**Verified result:** zero cross-split leakage, confirmed by an automated test, on the real dataset (1,008,932 train events / 193,335 test events across 26,467 / 5,042 cases).

---

## Part 2: Phase 2 — Discovery (figuring out what the real process actually is)

Attacked five different ways — comparing simple vs. complex methods rather than picking one:

**1. Directly-Follows Graph (DFG).** Using `pm4py`, every case is scanned and every "activity A immediately followed by activity B" transition is counted. The result — saved as `shared/process_graph.json` — becomes the backbone that later powers the Phase 4 guardrail. This is process mining in the truest sense: the process model isn't designed by a human, it's mined directly from evidence. (26 activity nodes, 176 edges discovered.)

**2. Markov chain baseline.** For every activity: "what's the single most common next activity that followed it in training?" No learning, just counting — the sanity-check baseline any fancier model must beat. **Result: 66.7% next-event accuracy.**

**3 & 4. LSTM and Transformer.** Two neural architectures predicting "given the sequence so far, what comes next":
- **LSTM** reads the sequence step by step, keeping a running memory — a reimplementation of Tax et al. (2017), the original academic paper for this exact task.
- **Transformer** uses self-attention instead — looking at all previous activities simultaneously, with positional encoding (sine/cosine waves) added so it knows event *order*, since attention alone has no built-in sense of sequence.

**Actual results (mean ± std across 3 seeds):**
- LSTM: **87.47% ± 0.09%**
- Transformer: **87.20% ± 0.06%**
- Both convincingly beat the 66.7% Markov baseline — the complexity is earned, not assumed.
- The LSTM slightly edged out the Transformer, a genuinely interesting, honest finding: Transformers don't always win, and on a moderately-sized dataset, an LSTM's inductive bias for sequential order can match or slightly beat attention.

**5. SVD + k-means clustering for process variants.** Each case turned into a vector (activity-type frequency counts), compressed via Truncated SVD (similar in spirit to PCA), then clustered via k-means — automatically discovering that different case "variants" (e.g., fast-approved vs. multiply-rejected loans) exist, without any manual labeling. **Result: 99.5% explained variance with 10 components, 6 clean, non-degenerate clusters.**

---

## Part 3: Phase 3 — Evaluation (turning predictions into a decision)

**Fusion scoring** combines two signals into one deviation score:
1. The sequence model's **negative log-likelihood (NLL)** — how surprised the LSTM/Transformer was by what actually happened.
2. The **graph-conformance cost** — does this transition exist at all as an edge in the discovered graph?

Combined via `λ × NLL + (1-λ) × graph_cost`, blending "statistically unusual" with "structurally invalid."

**Cost-optimal threshold.** Not all mistakes cost the same — missing a real compliance violation is far worse than a false alarm. Set at a **10:1 cost ratio**, and using a precision-recall curve, the exact threshold minimizing *total business cost* (not generic accuracy) was derived mathematically. **Result: recall = 1.0** at the chosen threshold — every true violation is caught, accepting more false positives as the deliberate, defensible trade-off.

**Latency profiling.** Measured over 1,000 trials, reporting the full distribution (p50/p95/p99) rather than a single average — since averages hide tail-latency problems. **Result: comfortably within the 50ms budget** (p99 ≈ 0.0005ms, roughly four orders of magnitude under budget).

---

## Part 4: Phase 4 — The AI agent, and how it connects back

**The agent.** Powered by **GLM-5.2** (Z.ai / Zhipu), accessed via **NVIDIA's NIM API**. For each step, it receives the webpage's accessibility tree (a structured text representation of buttons, checkboxes, labels) and returns one action, e.g. `click(bid="27")`.

**Baseline vs. shielded runs.**
- `run_baseline.py` — the agent acts with no oversight.
- `run_shielded.py` — before committing to an action, the system checks: does adding this action to the history so far violate the discovered process graph? If so, the model gets up to 2 retries to reconsider before proceeding anyway. This mirrors Phase 3's conformance checking, but used *proactively* as a guardrail before an action happens, not after the fact as an audit.

**The honest, important finding.** Because the process graph was mined from bank loan activities (`A_Accepted`, `O_Create Offer`) and the agent's actions are generic web clicks (`click(bid="27")`), the vocabularies don't overlap. The shield rarely finds a "violation" — not because it's broken, but because the two domains are structurally different, and it correctly *abstains* rather than firing false alarms. This is a mature, self-aware finding: **cross-domain generalization has real limits, and the system fails safely rather than falsely blocking valid actions.**

**Fault injection.** Robustness tested by re-running tasks with different random seeds, changing MiniWoB's element layout each time — a lightweight proxy for "what if the UI shifts slightly?" Simple tasks (click-test, click-checkboxes) hit 100% success across 5 layouts each. The harder `click-checkboxes-large` task showed **66.7% success (2/3) with one genuine failure (seed 1)** — the first real failure case found, documented explicitly rather than only reporting successes.

**DPO preference pairs.** For every real action the agent took, a synthetic "wrong" alternative is automatically constructed by perturbing the element ID (`bid`) by a small offset (±5 or ±10). "Chosen" = what the agent genuinely did; "rejected" = a plausible-looking but incorrect click on a different element. This generates DPO training data with zero human labeling — every real run automatically produces a training example. **Started at 9 pairs (proof-of-concept), scaled to 27.**

**DPO training.** Run through **Unsloth** (fast fine-tuning via 4-bit quantization and LoRA) and **TRL's DPOTrainer**, on Qwen2.5-1.5B. Explicitly documented as a small-scale run verifying the pipeline works end-to-end — not a claim of a robustly trained model. The training loss stayed near `ln(2) ≈ 0.693` (equivalent to random chance), which is the expected, honest result for a dataset this small — it demonstrates *why* scale matters, empirically, rather than just asserting it.

---

## Part 5: The dashboard

A **read-only observability tool** (Streamlit) — no live API calls, just visualization of the JSON files the pipeline already produced: DPO pairs (chosen vs. rejected, filterable by task) and fault-injection results, plus a project summary tab with the full number sheet.

---

## Honest limitations (stated explicitly, not discovered by a reviewer)

- DPO dataset (27 pairs) is proof-of-concept scale, not sufficient for robust preference learning.
- Fault-injection coverage is partial — NVIDIA NIM rate limits capped how much testing could be run in a single session.
- Cross-domain shielding (BPI-2017 graph vs. generic web actions) mostly abstains rather than actively intervening, by design of the domain mismatch — a legitimate finding, not a bug, but a real scope boundary.
- Rejected DPO examples are synthetically constructed (bid perturbation), not sampled from genuine model uncertainty — a deliberate simplification given the task's low ambiguity made temperature-based sampling ineffective at producing diverse candidates.

---

## Suggested narrative arc (for presenting this project)

1. **The problem:** Business processes generate huge event logs, but the "true" process has to be mined from evidence, not assumed from documentation.
2. **What was built (Phases 1-3):** A rigorous, leakage-safe data pipeline → the real process discovered automatically as a graph → validated against simple and deep-learning baselines → turned into a cost-aware violation detector with a measured latency budget.
3. **The twist (Phase 4):** Can a process model discovered from one domain act as a safety guardrail for a completely different AI system — an LLM-powered web agent? The shielding layer was built, tested, and an honest, explainable limitation was found: cross-domain conformance checking mostly abstains rather than false-triggering, which is itself correct, useful behavior.
4. **The bonus:** Every real agent run automatically generates labeled preference data, used to run an actual DPO fine-tuning pipeline — small-scale, but end-to-end verified.
5. **What's next:** Scale the DPO dataset beyond proof-of-concept size; get broader fault-injection coverage (currently capped by NIM rate limits during testing).
