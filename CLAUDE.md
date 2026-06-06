# CLAUDE.md

Guidance for Claude Code working in this repository. This file is the top-level
map; deeper technical detail lives in `vault/` (linked per section).

## Project Overview
Sparse shallow-network framework for solving Hamilton-Jacobi-Bellman (HJB)
equations from value/gradient samples, using a primal-dual proximal algorithm
(PDAP) with non-convex regularization and a semismooth Newton (SSN) outer solve.
Ports and extends a MATLAB reference at `/Users/ruizhechao/Documents/NonConvexSparseNN/`.

## References
Thesis & papers: `/Users/chaoruiz/Documents/NotePaper/MasterThesis/`
- **Non-convex regularization** (main): `nonconvexity/noncovex regulerization.pdf`
- **Gradient-augmented regression** (main): `PDE/Gradient-Augmented Regression.pdf`
- Supporting: `nonconvexity/` (integral representation of convex functions on
  measures, l.s.c. of the regularizer, relaxation of non-convex functionals),
  `PDE/` (Barron, char_HJB).


## Directory Notes
- `src/` — active code; `scripts/` — experiment runners & dataset generators (index in `vault/algorithm.md`).
- `experiments/` — curated experiment definitions, results, and promoted figures.
- `notebook/` — curated explanatory notebooks for SSN, PDAP configuration, and selected result views.
- `docs/research-directions.md` — restored provenance log for the research-direction summaries.
- `autoresearch/` — legacy research outputs; migrate from here into curated experiment docs before adding new summaries.
- `vault/` — layered technical docs (this file links them).
- `rawdata/` — generated datasets/plots; `outdated/` — deprecated experiments.
- `CONTEXT.md` — domain language / example-selection notes for the open-loop study.
