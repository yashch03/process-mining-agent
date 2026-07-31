"""
Observability dashboard — Section 8.13 of the implementation doc.
Read-only visualization of saved DPO pairs and fault-injection results.
No live API calls; reads directly from committed JSON files.
"""
import json
import os
import streamlit as st

st.set_page_config(page_title="Process-Mining-Agent Dashboard", layout="wide")
st.title("Conformance-Guided Agent — Run Inspector")

# Locate repo root regardless of where streamlit is launched from
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DPO_PAIRS_PATH = os.path.join(REPO_ROOT, "phase4_agent", "dpo", "preference_pairs.json")
FAULT_INJECTION_DIR = os.path.join(REPO_ROOT, "phase4_agent", "fault_injection")


def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


tab1, tab2, tab3 = st.tabs(["DPO Preference Pairs", "Fault Injection Results", "Project Summary"])

with tab1:
    st.header("DPO Preference Pairs")
    pairs = load_json_safe(DPO_PAIRS_PATH)

    if pairs is None:
        st.warning(f"No file found at {DPO_PAIRS_PATH}")
    else:
        st.metric("Total pairs", len(pairs))

        tasks = sorted(set(p["task"] for p in pairs))
        selected_task = st.selectbox("Filter by task", ["All"] + tasks)

        filtered = pairs if selected_task == "All" else [p for p in pairs if p["task"] == selected_task]
        st.write(f"Showing {len(filtered)} pairs")

        for i, pair in enumerate(filtered):
            with st.expander(f"Pair {i+1}: {pair['context'][:60]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Chosen:** `{pair['chosen']}`")
                with col2:
                    st.error(f"**Rejected:** `{pair['rejected']}`")
                st.caption(f"Task: {pair['task']}")

with tab2:
    st.header("Fault Injection Results")

    if not os.path.isdir(FAULT_INJECTION_DIR):
        st.warning(f"No directory found at {FAULT_INJECTION_DIR}")
    else:
        result_files = [f for f in os.listdir(FAULT_INJECTION_DIR) if f.startswith("results_") and f.endswith(".json")]

        if not result_files:
            st.info("No fault-injection result files found yet.")
        else:
            for fname in result_files:
                data = load_json_safe(os.path.join(FAULT_INJECTION_DIR, fname))
                if data:
                    st.subheader(data.get("task", fname))
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Success rate", f"{data.get('success_rate', 0):.2%}")
                    col2.metric("Seeds tested", len(data.get("seeds_tested", [])))
                    col3.metric("Failures", data.get("rewards", []).count(0))

                    st.write("**Per-seed results:**")
                    for seed, reward in zip(data.get("seeds_tested", []), data.get("rewards", [])):
                        status = "✅ Success" if reward > 0 else "❌ Failed"
                        st.write(f"Seed {seed}: {status} (reward={reward})")

                    if data.get("note"):
                        st.caption(f"Note: {data['note']}")

with tab3:
    st.header("Project Summary")
    st.markdown("""
    **Phase 1-3 (Person A):** Real ingestion, temporal split, DFG extraction,
    Markov baseline (0.667), SVD clustering, LSTM (0.8747±0.0009) and
    Transformer (0.8720±0.0006) — both beat the baseline, cost-asymmetric
    fusion/threshold (recall=1.0 under 10:1 cost matrix), latency well
    within the 50ms budget.

    **Phase 4 (Person B):** BrowserGym + shielding logic verified against
    the real process graph, working NIM-backed agent, multi-task
    shielded-vs-baseline evaluation, DPO preference pairs (27, scaled from
    an initial proof-of-concept of 9), DPO training pipeline verified
    end-to-end with Unsloth + TRL.

    **Honest limitations:** DPO dataset is still small-scale relative to
    what would be needed for robust preference learning. Fault-injection
    coverage is partial due to NIM rate limits encountered during testing.
    """)
