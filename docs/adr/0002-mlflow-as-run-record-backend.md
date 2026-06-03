---
status: proposed
---

# MLflow is a backend behind the Run Record interface, not a parallel tracker

We will expose MLflow as a swappable *backend* behind the existing
`ExperimentRun` / `RunRecordWriter` Run Record interface (`src/experiment_logging.py`),
selected by environment: with `MLFLOW_TRACKING_URI` set, a Run Record will be
logged to MLflow; unset, it will be written as local JSON (the current, and
current-default, behavior). One interface, two adapters. The seam itself is
unchanged, though the runners need one small change — persisting the full result
for local curve storage — described under "What MLflow stores" below. We chose
this over rewriting the legacy `src/mlflow_utils.py`, over making MLflow the sole
store, and over running both stores in parallel.

**Implementation status: proposed, not yet built.** As of this ADR,
`src/experiment_logging.py` always writes local JSON and has no
`MLFLOW_TRACKING_URI` branch or MLflow adapter; `src/mlflow_utils.py` and
`notebook/pdpa_vdp.ipynb` still exist unchanged. Everything below describes the
intended design, not current behavior. Flip this ADR to `accepted` when the
adapter lands.

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
  hierarchies; it is to be superseded. The MLflow adapter will log from the same
  flat summary `RunRecordWriter` consumes, not from PDAP internals.

## Consequences

- Once built, switching local-vs-remote will be a pure environment change; no
  code edits and no changes to how experiments are launched.
- The vocabulary is fixed by the glossary (`CONTEXT.md`): an MLflow run *is* a
  **Run Record** and an MLflow file *is* a **Run Artifact**. The
  local-JSON-vs-MLflow split is a backend detail and is deliberately kept out of
  the glossary.

## What MLflow stores, and what stays local

The intended use of MLflow is **a cross-config comparison dashboard**, not a
training monitor. That fixes what each side holds:

- **MLflow holds the matrix, summary-only.** One run per *config point*
  `(activation, power, loss, use_sphere, seed, gamma, …)` — flat, not nested.
  Gamma is an ordinary param, not a special axis or a nesting level. Every
  dimension is logged as a **param**; the per-point summary scalars (`h1`, `n`,
  `score`, the `best_*` fields) are logged as **metrics**. The comparison views
  (runs table, parallel-coordinates, scatter) are chosen in the MLflow UI at
  look-time, so logging richly keeps every view open. This is still logged from
  the flat summary — the `per_gamma` rows of one Run Record fan out into one
  MLflow run per gamma point — so the "log from the summary, not PDAP internals"
  rule **survives** (it would only have reversed if curves went into MLflow).
- **Per-iteration training curves stay local.** They are deliberately *not*
  logged to MLflow. The full `PDAP.fit()` result (loss/error curves, weights) is
  saved as a sibling `result_<run_id>.pkl` Run Artifact on disk and loaded only
  when a single run is inspected. MLflow records the pickle's filename as a
  **tag** — a pointer to the local file, not the data.

### Why curves are not in MLflow

A reasonable reader will ask why the loss curves aren't in the tracking UI. The
dashboard's job here is cross-config comparison, which the summary scalars
cover; per-iteration series are needed only when drilling into a single run,
where loading the local pickle is fine. Putting them in MLflow would force the
adapter to read raw PDAP per-iteration internals — reversing the "log from the
summary" rule for data that does not need to be central. Keeping curves local
keeps the adapter thin and the MLflow store light.

## Planned follow-up work (not done yet)

- In the runners, persist the full `PDAP.fit()` result as
  `result_<run_id>.pkl` per config point. **Currently the curves are discarded**
  after `best_iteration`, so without this there is nothing to drill into.
- Add the MLflow adapter + `MLFLOW_TRACKING_URI` backend switch to
  `src/experiment_logging.py`: one run per config point (gamma as an ordinary
  param), summary scalars as metrics, the pickle filename as a tag — logged from
  the flat summary.
- Retire `src/mlflow_utils.py` and update the `notebook/pdpa_vdp.ipynb` call
  site once the adapter is proven.
- Optional one-off: backfill pre-existing `models/*.pkl` results into MLflow via
  a throwaway "JSON records → MLflow" script (not a maintained code path).
