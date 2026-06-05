---
status: accepted
---

# Experiments are curated definitions, not raw output folders

Curated experiment definitions will live under `experiments/<experiment_name>/`
instead of expanding the legacy `autoresearch/` tree. Each experiment directory
owns the research question, canonical command, analysis script, promoted final
figures, and Markdown summaries. Executable experiment config lives in
`conf/experiment/<experiment_name>.yaml` so Hydra composition remains centralized.
Raw datasets, logs, run records, Hydra output, and intermediate plots stay under
`rawdata/` using the central paths in `src.paths`.

Top-level `scripts/` is reserved for generic reusable entry points.
Domain-specific runners may remain in `scripts/` while they are legacy and
unclassified, but new curated experiments should have an
`experiments/<name>/run.py` orchestrator. The experiment orchestrator owns sweep
policy, repeated calls, aggregation triggers, and output layout, but it should
use the existing PDAP/config interface (`PDAP.from_config(...)` and
`fit_from_config(...)`) plus shared data, logging, and plotting utilities rather
than reimplementing PDAP training or adding a second training interface.

The generic runner boundary is deliberately narrow: run one independent PDAP
training config point on one dataset, compose the configured PDAP model, train
it, write a Run Record, and save only minimal generic artifacts such as a config
snapshot, final metrics, and the full fit result needed for drill-down analysis.
Following ADR-0002, the full fit result is a Run Artifact referenced by the Run
Record, not embedded in the Run Record JSON.
All experiment data loading is centralized around the shared `(x, V(x), dV(x))`
value-sample contract; experiments select datasets rather than defining custom
loaders.
Generic PDAP training normalizes value samples by default as pre-training data
preprocessing, not as optimizer behavior. The value-sample loader and reversible
normalization transform should live in `src/data.py`, and the fitted transform
must be recorded in the Run Record or artifacts; disabling normalization is a
diagnostic or legacy-reproduction choice, not the default experiment path.
Performance-evaluation functions that become reusable should live in `src`, but
the current model-level relative errors may remain in the model/PDAP training
path until there is a concrete extraction need. Experiments select,
parameterize, aggregate, and present metrics; table helpers such as
`src/metric.py` remain reporting utilities, not the owner of performance
evaluation. Plot function implementations are centralized in `src/plots.py`;
experiments may call those functions but should not own local plotting logic.
Normalization, multi-dataset comparisons, plot invocation, report generation,
and search policy beyond Hydra multirun stay outside the generic runner until
repeated needs justify promotion.

Curated notebooks are limited to three roles: `ssn_optimizer.ipynb` for the SSN
optimizer interface, `pdap_model_configuration.ipynb` for configuring runnable
PDAP models, and `experiment_results_<name>.ipynb` for selected result views
that read Run Records and artifacts. Legacy notebooks are not the source of
truth for sweeps or experiments; they should either be retired or migrated into
an experiment directory after classification.

Final result presentation is Markdown-first: each curated experiment should
publish its stable read in `results.md` or `README.md` with promoted final
figures, while notebooks remain optional interactive views over the same Run
Records and artifacts.

The legacy VDP notebooks classify into separate experiments: `pdpa_vdp.ipynb`
belongs to `activationsearch`, while `pdpa_v3_vdp.ipynb` belongs to
`penaltypowers`. `experiment_analysis.ipynb` is still unclassified.

We chose this over making notebooks the experiment source of truth, over
continuing to grow `autoresearch/`, and over putting every runner in top-level
`scripts/`. The trade-off is a later migration step for legacy outputs and
runners, but the benefit is a clear separation between curated research
definitions, reusable execution machinery, and disposable local artifacts.

The current `scripts/train.py` name is provisional: its role is narrower than
generic "training" and should be reconsidered after the experiment orchestrator
shape is proven. It may be retired entirely or renamed to a one-off PDAP config
runner if curated experiment runners become the normal execution path.
