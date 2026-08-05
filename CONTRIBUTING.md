# Contributing / Development Workflow

This was built by a 2-person team under a compressed, free-tier-only sprint.
Documenting the actual workflow used, for anyone extending this project.

## Environment split
- Phase 1-3 (ingestion, discovery, evaluation): developed on Lightning AI
  (CPU Studio) + Kaggle (GPU, for model training).
- Phase 4 (agent, shielding, DPO): developed on Lightning AI (CPU Studio,
  browser automation) + Kaggle (GPU, for DPO training).

## Interface contract
`shared/process_graph.json` is the handoff point between Phase 2 (produces it)
and Phase 4 (consumes it for shielding). Phase 4 development used a synthetic
placeholder matching this schema until the real file was available, enabling
fully parallel work.

## Before committing
Always verify file content with `cat`, not just `wc -l` — several files in
this project's history were committed empty due to terminal/session issues
and only caught by explicit content verification.

## Running tests
```bash
pytest tests/ -v
```

## Known environment gotchas
See `phase4_agent/browsergym_env/SETUP_NOTES.md` and
`dashboard/SETUP_NOTES.md` for documented fixes to real issues hit during
development (Ubuntu package naming, MiniWoB file persistence, Streamlit
behind Lightning's proxy).
