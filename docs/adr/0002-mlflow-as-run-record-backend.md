---
status: accepted
---

# MLflow is a backend behind the Run Record interface, not a parallel tracker

We will expose MLflow as a swappable *backend* behind the existing
`ExperimentRun` / `RunRecordWriter` Run Record interface (`src/experiment_logging.py`),
selected by environment: with `MLFLOW_TRACKING_URI` set, a Run Record is logged
to MLflow; unset, it is written as local JSON (the current default). One
interface, two adapters; runner scripts and `RunRecordWriter` are unchanged. We
chose this over rewriting the legacy `src/mlflow_utils.py`, over making MLflow
the sole store, and over running both stores in parallel.

## Considered options

- **A standalone MLflow wrapper / generic logging core** — rejected: MLflow's
  fluent API is already a generic logging layer, so a hand-rolled core over it
  would be a shallow module. (This reverses an earlier inclination to keep MLflow
  separate with "no abstraction" — that reasoning is moot now that the Run Record
  interface already exists for its own, MLflow-independent reasons.)
- **MLflow replaces the local JSON writer** — rejected: it would discard the
  zero-dependency, offline, debuggable local default that the Run Record system
  was deliberately built around.
- **Local JSON and MLflow always written in parallel** — rejected: double
  bookkeeping on every run and two stores that can silently drift.
- **Keep the legacy `mlflow_utils.py` independent** — rejected: it logs the raw
  `PDAP.fit()` result with duplicated metric loops and two inconsistent run
  hierarchies; it is superseded and will be retired. The MLflow adapter logs from
  the same flat summary `RunRecordWriter` consumes, not from PDAP internals.

## Consequences

- Switching local-vs-remote is a pure environment change; no code edits and no
  changes to how experiments are launched.
- The vocabulary is fixed by the glossary (`CONTEXT.md`): an MLflow run *is* a
  **Run Record** and an MLflow file *is* a **Run Artifact**. The
  local-JSON-vs-MLflow split is a backend detail and is deliberately kept out of
  the glossary.
- `src/mlflow_utils.py` and its notebook call site (`notebook/pdpa_vdp.ipynb`)
  are retired as part of this work; backfilling pre-existing `models/*.pkl`
  results, if wanted, becomes a one-off "JSON records → MLflow" script rather than
  a maintained code path.
