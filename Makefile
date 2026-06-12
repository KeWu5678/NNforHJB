PY := .venv/bin/python
TF_DIR ?= deploy/terraform
MLFLOW_RECORDS ?= rawdata/logs/multirun

# This root Makefile is the project's experiment entrypoint.
#
# Convention:
#   - Add one public target per curated experiment, e.g. `activationsearch_VDP`.
#   - Keep the experiment's config in `conf/experiment/<name>.yaml`.
#   - Keep experiment-owned analysis/results under `experiments/<name>/`.
#   - Use `scripts/train.py` only for a single Hydra-composed training run.
#
# If this file grows too large, split target bodies into included fragments such
# as `experiments/<name>/experiment.mk`; keep the root Makefile as the discoverable
# interface for humans and CI.
#
# Per-run logging. Quiet by default; pass `VERBOSE=true` to print each run's
# PDAP progress tables to the console, e.g. `make activationsearch_VDP VERBOSE=true`.
VERBOSE ?= false
# Parallelism for the multirun sweeps (Hydra joblib launcher). JOBS runs are
# launched at once; each is pinned to one BLAS thread (OMP_NUM_THREADS=1 in the
# recipes) so the workers don't oversubscribe the cores. Override per-invocation,
# e.g. `make activationsearch_pendulum JOBS=10`; JOBS=1 is effectively serial.
JOBS ?= 8
.PHONY: help activationsearch_VDP activationsearch_pendulum penaltypowers_VDP mlflow-deploy mlflow-start mlflow-stop mlflow-tunnel mlflow-backfill mlflow-backfill-latest mlflow-backfill-dry-run mlflow-backfill-latest-dry-run

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

mlflow-backfill-latest:  ## upload only the newest local Run Record JSON files to MLflow
	$(PY) scripts/upload_run_records_to_mlflow.py $(MLFLOW_RECORDS) --latest

mlflow-backfill-dry-run:  ## list local Run Record JSON files that would be uploaded
	$(PY) scripts/upload_run_records_to_mlflow.py $(MLFLOW_RECORDS) --dry-run

mlflow-backfill-latest-dry-run:  ## list newest local Run Record JSON files that would be uploaded
	$(PY) scripts/upload_run_records_to_mlflow.py $(MLFLOW_RECORDS) --latest --dry-run

activationsearch_VDP:  ## reproduce: two Hydra multirun sweeps (signed/profile + semiconcave/finite_step, VDP) + analysis
	# Sweep 1: signed / profile
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=activationsearch_VDP \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=rawdata/logs/multirun/activationsearch_VDP/signed_profile \
	  env.verbose=$(VERBOSE) \
	  env.seed=42 \
	  model.activation=tanh,softplus,matern52,gaussian \
	  model.gamma=0,0.1,1,10 \
	  'model.loss_weights=[1.0,0.0],[1.0,1.0]'
	# Sweep 2: semiconcave / finite_step
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=activationsearch_VDP \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=rawdata/logs/multirun/activationsearch_VDP/semiconcave_finitestep \
	  env.verbose=$(VERBOSE) \
	  env.seed=42 \
	  model.kind=semiconcave model.insertion=finite_step \
	  model.activation=tanh,softplus,matern52,gaussian \
	  model.gamma=0,0.1,1,10 \
	  'model.loss_weights=[1.0,0.0],[1.0,1.0]'
	$(PY) experiments/activationsearch_VDP/analysis.py


activationsearch_pendulum:  ## reproduce: two Hydra multirun sweeps (signed/profile + semiconcave/finite_step, pendulum) + analysis
	# Sweep 1: signed / profile
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=activationsearch_pendulum \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=rawdata/logs/multirun/activationsearch_pendulum/signed_profile \
	  env.verbose=$(VERBOSE) \
	  env.seed=42 \
	  model.activation=tanh,softplus,matern52,gaussian \
	  model.gamma=0,0.1,1,10 \
	  'model.loss_weights=[1.0,0.0],[1.0,1.0]'
	# Sweep 2: semiconcave / finite_step
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=activationsearch_pendulum \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=rawdata/logs/multirun/activationsearch_pendulum/semiconcave_finitestep \
	  env.verbose=$(VERBOSE) \
	  env.seed=42 \
	  model.kind=semiconcave model.insertion=finite_step \
	  model.activation=tanh,softplus,matern52,gaussian \
	  model.gamma=0,0.1,1,10 \
	  'model.loss_weights=[1.0,0.0],[1.0,1.0]'
	$(PY) experiments/activationsearch_pendulum/analysis.py


penaltypowers_VDP:  ## reproduce: finite-step penalty sweep (VDP) + analysis
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=penaltypowers_VDP \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=rawdata/logs/multirun/penaltypowers_VDP \
	  env.verbose=$(VERBOSE) \
	  env.seed=42 \
	  model.power=2.0,2.01,3.0,4.0,5.0 \
	  model.gamma=0,0.01,0.1,1,10 \
	  'model.loss_weights=[1.0,0.0],[1.0,1.0]'
	$(PY) experiments/penaltypowers_VDP/analysis.py
