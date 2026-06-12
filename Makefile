PY := .venv/bin/python
TF_DIR ?= deploy/terraform
MLFLOW_RECORDS ?= rawdata/logs/multirun

# This root Makefile is the project's experiment entrypoint.
#
# Convention:
#   - Add one public target per curated experiment, e.g. `activationsearch`.
#   - Keep the experiment's config in `conf/experiment/<name>.yaml`.
#   - Keep experiment-owned analysis/results under `experiments/<name>/`.
#   - Use `scripts/train.py` only for a single Hydra-composed training run.
#
# If this file grows too large, split target bodies into included fragments such
# as `experiments/<name>/experiment.mk`; keep the root Makefile as the discoverable
# interface for humans and CI.
#
# Per-run logging. Quiet by default; pass `VERBOSE=true` to print each run's
# PDAP progress tables to the console, e.g. `make activationsearch VERBOSE=true`.
VERBOSE ?= false
# Parallelism for the multirun sweeps (Hydra joblib launcher). JOBS runs are
# launched at once; each is pinned to one BLAS thread (OMP_NUM_THREADS=1 in the
# recipes) so the workers don't oversubscribe the cores. Override per-invocation,
# e.g. `make activationsearch JOBS=10`; JOBS=1 is effectively serial.
JOBS ?= 8
HOMOGENEOUS_ACTIVATIONS = $(shell $(PY) -c 'from src.config.activations import ACTIVATIONS; print(",".join(name for name, (_, use_sphere) in ACTIVATIONS.items() if use_sphere))')
.PHONY: help activationsearch region_split_pendulum penaltypowers mlflow-deploy mlflow-start mlflow-stop mlflow-tunnel mlflow-backfill mlflow-backfill-latest mlflow-backfill-dry-run mlflow-backfill-latest-dry-run

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "} {printf "  %-20s %s\n", $$1, $$2}'
	@printf "\n  %-20s %s\n" "VERBOSE=true" "also stream PDAP tables to console (always in per-run run.log)"
	@printf "  %-20s %s\n" "JOBS=N" "parallel sweep workers (default 8)"
	@printf "  %-20s %s\n" "MLFLOW_RECORDS=PATH" "JSON record file/dir for backfill (default rawdata/logs/multirun)"
	@printf "  %-20s %s\n" "MLFLOW_TRACKING_URI=URL" "set in your shell to enable MLflow logging/backfill"

mlflow-deploy:  ## provision/update EC2 MLflow tracking server with Terraform
	terraform -chdir=$(TF_DIR) init
	terraform -chdir=$(TF_DIR) apply

mlflow-start:  ## start the MLflow EC2 instance; systemd starts the server on boot
	aws ec2 start-instances --instance-ids "$$(terraform -chdir=$(TF_DIR) output -raw instance_id)"

mlflow-stop:  ## stop the MLflow EC2 instance to avoid compute charges
	aws ec2 stop-instances --instance-ids "$$(terraform -chdir=$(TF_DIR) output -raw instance_id)"

mlflow-tunnel:  ## open SSM tunnel; keep this terminal running
	eval "$$(terraform -chdir=$(TF_DIR) output -raw ssm_port_forward_command)"

mlflow-backfill:  ## upload existing local Run Record JSON files to MLflow
	$(PY) scripts/upload_run_records_to_mlflow.py $(MLFLOW_RECORDS)

mlflow-backfill-latest:  ## upload only the latest full sweep's Run Record JSON files to MLflow
	$(PY) scripts/upload_run_records_to_mlflow.py $(MLFLOW_RECORDS) --latest-run

mlflow-backfill-dry-run:  ## list local Run Record JSON files that would be uploaded
	$(PY) scripts/upload_run_records_to_mlflow.py $(MLFLOW_RECORDS) --dry-run

mlflow-backfill-latest-dry-run:  ## list the latest full sweep's Run Record JSON files that would be uploaded
	$(PY) scripts/upload_run_records_to_mlflow.py $(MLFLOW_RECORDS) --latest-run --dry-run

activationsearch:  ## sweep: data × kind × insertion × activation × alpha × gamma × loss_weights → results.md
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=activationsearch \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=rawdata/logs/multirun/activationsearch \
	  env.verbose=$(VERBOSE) \
	  env.seed=42 \
	  data=vdp,pendulum \
	  model.kind=signed,semiconcave \
	  model.insertion=profile,finite_step \
	  model.activation=tanh,softplus,matern52,gaussian,gelu_squared,silu_squared,rcip_2,snake_b0_25,lisht,gausscent_1 \
	  model.alpha=1e-2,1e-3,1e-4,1e-5 \
	  model.gamma=0,0.1,1,10 \
	  'model.loss_weights=[1.0,0.0],[1.0,1.0]'
	$(PY) experiments/activationsearch/analysis.py

region_split_pendulum:  ## sweep: kind × activation × gamma on pendulum → smooth-vs-switching H1 table
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=region_split_pendulum \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=rawdata/logs/multirun/region_split_pendulum \
	  env.verbose=$(VERBOSE) \
	  env.seed=42 \
	  data=pendulum \
	  model.kind=signed,semiconcave \
	  model.insertion=profile \
	  model.activation=tanh,softplus,matern52,gaussian,gelu_squared \
	  model.gamma=0,1 \
	  'model.loss_weights=[1.0,1.0]'
	$(PY) experiments/region_split_pendulum/analysis.py


penaltypowers:  ## sweep: data × homogeneous activation × penalty power × gamma × loss_weights → results.md
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=penaltypowers \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=rawdata/logs/multirun/penaltypowers \
	  env.verbose=$(VERBOSE) \
	  env.seed=42 \
	  data=vdp,pendulum \
	  model.activation=$(HOMOGENEOUS_ACTIVATIONS) \
	  model.power=2.0,2.01,3.0,4.0,5.0 \
	  model.gamma=0,0.01,0.1,1,10 \
	  'model.loss_weights=[1.0,0.0],[1.0,1.0]'
	$(PY) experiments/penaltypowers/analysis.py
