# SparseNNforHJB

## MLflow Pipeline

`ExperimentRun` always writes a local JSON Run Record in the Hydra output
directory. When `MLFLOW_TRACKING_URI` is set, the same completed record is also
uploaded to MLflow for dashboard comparison. MLflow stores params, scalar
metrics, status, Hydra metadata, and local artifact paths; it does not upload
the `result_<run_id>.pkl` artifact in the current pipeline.

Start or connect to an MLflow tracking server first.

For the EC2/Terraform deployment, provision once:

```bash
make mlflow-deploy
```

### Runtime logging

Then start the instance when needed and open the SSM tunnel in a dedicated
terminal:

```bash
make mlflow-start
make mlflow-tunnel
```

`mlflow-tunnel` keeps running while the dashboard is connected. The EC2 instance
starts the MLflow server through systemd; you do not manually run `mlflow server`
on your laptop for this path.

For a purely local server instead:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Once the tracking server is reachable, set:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Run a curated experiment through the Makefile:

```bash
make activationsearch
```

Or run the lower-level Hydra training entrypoint directly:

```bash
uv run python scripts/train.py +experiment=activationsearch data=pendulum
```

At `run.finish()`, the script writes the local JSON Run Record, keeps it on disk,
and publishes dashboard data to MLflow. If `MLFLOW_TRACKING_URI` is unset, the
run is local-only. Project run IDs use:

```text
{experiment_name}_{data_choice}_{YYYYMMDD}_{4hex}
```

### Backfill existing records

Existing JSON Run Records under `rawdata/logs/multirun` can be uploaded to
MLflow without rerunning training. You still need a reachable MLflow tracking
server. Provision with `make mlflow-deploy` only if the MLflow EC2 infrastructure
does not exist yet.

For the EC2 server, boot the instance and open the tunnel first (same as a live
run):

```bash
make mlflow-start
make mlflow-tunnel   # keep running in a dedicated terminal
```

For a local server, just have `mlflow server` running instead — no start/tunnel
needed. Then point at the tracking server and backfill:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
make mlflow-backfill
```

For a sweep directory such as `rawdata/logs/multirun/activationsearch`, this
uploads every JSON record in that directory — including stale job dirs left over
from earlier sweeps. To upload only the **latest full sweep**, use
`mlflow-backfill-latest`. It finds the sweep's `multirun.yaml` launch marker
(Hydra rewrites it on every sweep) and uploads only the records written at or
after that marker, dropping leftovers from previous sweeps:

```bash
make mlflow-backfill-latest MLFLOW_RECORDS=rawdata/logs/multirun/activationsearch
```

Limit upload to one experiment or one Hydra job directory:

```bash
make mlflow-backfill MLFLOW_RECORDS=rawdata/logs/multirun/activationsearch
make mlflow-backfill MLFLOW_RECORDS=rawdata/logs/multirun/activationsearch/1069
```

Preview what would be uploaded without calling MLflow:

```bash
make mlflow-backfill-dry-run
make mlflow-backfill-latest-dry-run MLFLOW_RECORDS=rawdata/logs/multirun/activationsearch
```

The importer reads each Run Record JSON and, for older records, enriches it with
adjacent `.hydra/overrides.yaml` metadata before uploading. Re-running the
importer creates additional MLflow runs for the same JSON; deduplication is not
implemented yet.

Stop the EC2 instance when idle:

```bash
make mlflow-stop
```

## PDPA_v1
### Phase 1: Inserting the new neurons

```insertion``` under the PDPA.py, it does the follwing things:
the function ```sample_uniform_sphere_points``` finds *** N *** initial value of the neurons.
the function ```local_maximize``` finds for each initial value, the local maximum of the funtion ```profile```, which calculates the dual profile for each to-be-inserted neuron.
