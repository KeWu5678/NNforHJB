PY := .venv/bin/python

# Per-run logging. Quiet by default (40 runs); `make activationsearch VERBOSE=true`
# prints each run's PDAP progress tables to the console.
VERBOSE ?= false

.PHONY: help activationsearch

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "} {printf "  %-20s %s\n", $$1, $$2}'

activationsearch:  ## reproduce: Hydra multirun sweep (signed/profile, VDP) + analysis
	$(PY) scripts/train.py -m +experiment=activationsearch \
	  hydra.sweep.dir=rawdata/logs/multirun/activationsearch \
	  env.verbose=$(VERBOSE) \
	  env.seed=42 \
	  model.activation=tanh,softplus,matern52,gaussian \
	  model.gamma=0,0.01,0.1,1,10 \
	  'model.loss_weights=[1.0,0.0],[1.0,1.0]'
	$(PY) experiments/activationsearch/analysis.py
