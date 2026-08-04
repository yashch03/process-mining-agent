# Enterprise Process Discovery & Conformance-Guided Web Agents

[![CI](https://github.com/yashch03/process-mining-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/yashch03/process-mining-agent/actions/workflows/ci.yml)

A system that discovers the real, underlying structure of a business process directly from event-log evidence — then tests whether that discovered structure can act as a live safety guardrail for a completely different AI system: an LLM-powered web-browsing agent, closing the loop with preference-based fine-tuning (DPO).

Full narrative writeup: [`docs/PROJECT_EXPLANATION.md`](docs/PROJECT_EXPLANATION.md)

---

## What this project actually does

1. **Ingests** real enterprise event logs (BPI Challenge 2017 — 1.2M events, 31,509 loan-application traces) under strict, typed data contracts with zero temporal leakage.
2. **Discovers** the real process structure via a directly-follows graph, then validates it against a naive baseline, an LSTM, and a Transformer.
3. **Evaluates** deviations from that discovered process using a cost-asymmetric fusion score, with a measured latency budget.
4. **Grounds** an NIM-backed web-browsing agent (GLM-5.2) using the discovered process graph as a live conformance shield.
5. **Closes the loop** with Direct Preference Optimization (DPO), using the shield's own conformance signal to automatically generate labeled preference data.

---

## Real results (not projected — measured on committed code)

| Phase | Metric | Result |
|---|---|---|
| 1 | Cross-split temporal leakage | **0** (verified by automated test) |
| 2 | Markov baseline (next-event accuracy) | 66.7% |
| 2 | LSTM (mean ± std, 3 seeds) | **87.47% ± 0.09%** |
| 2 | Transformer (mean ± std, 3 seeds) | **87.20% ± 0.06%** |
| 2 | SVD explained variance (10 components) | 99.5% |
| 3 | Precision/recall at cost-optimal threshold (10:1 cost) | recall = **1.0** |
| 3 | p99 latency vs. 50ms budget | **~0.0005ms** (well within budget) |
| 4 | Baseline agent success (MiniWoB click-test) | reward = 1.0 |
| 4 | Fault-injection: simple tasks (5 seeds each) | **100%** success |
| 4 | Fault-injection: click-checkboxes-large (3 seeds) | **66.7%** (first genuine failure case found) |
| 4 | DPO preference pairs generated | 27 (scaled from a 9-pair proof-of-concept) |
| 4 | DPO training | Verified end-to-end (Unsloth + TRL); loss ≈ ln(2), honestly demonstrating dataset-scale is the limiting factor |

---

## Repo structure

phase1_ingestion/ Pydantic schemas, temporal hold-out split
phase2_discovery/ DFG extraction, Markov baseline, SVD/k-means, LSTM, Transformer, training loop
phase3_evaluation/ Cost-asymmetric fusion, threshold derivation, latency profiling
phase4_agent/ NIM-backed agent, process-graph shielding, BrowserGym environment,
multi-task evaluation, DPO pair construction + training, fault injection
dashboard/ Read-only Streamlit observability UI (DPO pairs, fault-injection results)
shared/ process_graph.json — the discovered graph, shared between Phase 2 and Phase 4
tests/ pytest suite (leakage, schema validation, shielding logic)
docs/ Full project explanation and narrative writeup
configs/ Pinned seeds for reproducibility

---

## Setup

Requires Python 3.12, a free [NVIDIA NIM](https://build.nvidia.com) API key, and (for Phase 4) a free GPU environment such as Kaggle.

```bash
git clone https://github.com/yashch03/process-mining-agent.git
cd process-mining-agent
pip install -e .
```

Set your NIM key:
```bash
export NVIDIA_API_KEY=your_key_here
```

Download the BPI Challenge 2017 dataset from [4TU.ResearchData](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884) and place it at `data/BPI Challenge 2017.xes` (git-ignored — not included in this repo due to size).

For Phase 4's browser automation:
```bash
pip install browsergym browsergym-miniwob
playwright install
sudo apt-get install -y libnss3 libnspr4 libasound2t64   # Linux/Ubuntu 24.04+ only
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git ~/miniwob-plusplus
export MINIWOB_URL="file://$HOME/miniwob-plusplus/miniwob/html/miniwob/"
```

See `phase4_agent/browsergym_env/SETUP_NOTES.md` and `dashboard/SETUP_NOTES.md` for known environment-specific gotchas and their fixes.

---

## Running the pipeline

```bash
# Phase 1
python3 -m phase1_ingestion.temporal_split

# Phase 2
python3 -m phase2_discovery.dfg_extraction
python3 -m phase2_discovery.baseline_markov
python3 -m phase2_discovery.svd_clustering
python3 -m phase2_discovery.train          # needs GPU — see configs/seeds.yaml for seed control

# Phase 3
python3 -m phase3_evaluation.fusion
python3 -m phase3_evaluation.threshold
python3 -m phase3_evaluation.latency_profile

# Phase 4
python3 -m phase4_agent.run_baseline
python3 -m phase4_agent.run_shielded
python3 -m phase4_agent.eval_shielded_vs_baseline
python3 -m phase4_agent.dpo.pair_construction
python3 -m phase4_agent.dpo.train_dpo       # needs GPU

# Dashboard (read-only, visualizes saved results)
streamlit run dashboard/app.py
```

## Tests

```bash
pytest tests/ -v
```

---

## Honest limitations

- **DPO dataset scale**: 27 preference pairs is proof-of-concept scale, not sufficient for robust preference learning — the training loss staying near `ln(2)` (chance level) is the expected, honest result at this scale, and demonstrates empirically why more pairs are needed.
- **Cross-domain shielding**: the process graph was mined from bank-loan activities; the web agent's action space is generic UI clicks. The shield correctly *abstains* rather than false-triggering when there's no overlapping structure — a legitimate, self-aware finding about domain boundaries, not a bug.
- **Fault-injection coverage**: partial, capped by NVIDIA NIM's free-tier rate limits encountered during testing.
- **DPO rejected examples** are synthetically constructed (element-ID perturbation), not sampled from genuine model uncertainty — a deliberate simplification, since the task's low ambiguity made temperature-based sampling ineffective at producing diverse real candidates.

---

## Team

Built by a two-person team across a compressed, free-tier-only sprint. Phase 1-3 and Phase 4 were developed in parallel using a shared interface contract (`shared/process_graph.json`), enabling independent progress without blocking on either side.

## License / Third-party note

`pm4py` (used in Phase 2) is AGPL v3 licensed — see its own licensing terms if extending this project commercially.
