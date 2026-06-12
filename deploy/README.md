# MLflow Deployment

Self-hosted MLflow tracking server on AWS for a central comparison dashboard.
Terraform provisions the infrastructure; EC2 `user_data` installs MLflow and
runs it under systemd. The root `Makefile` is the normal operator interface.

```
your machine --SSM tunnel--> EC2 (mlflow server, 127.0.0.1:5000)
                              └─ backend store: sqlite on EBS
client config: MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

The application currently logs params, scalar metrics, tags, and local artifact
paths. It does not upload `result_<run_id>.pkl` or JSON records as MLflow
artifacts. The Terraform module does not create an S3 artifact bucket in v1.

See:

- [`docs/adr/0001-self-hosted-mlflow-on-ec2.md`](../docs/adr/0001-self-hosted-mlflow-on-ec2.md)
- [`docs/adr/0002-mlflow-as-run-record-backend.md`](../docs/adr/0002-mlflow-as-run-record-backend.md)

## Prerequisites

- Terraform `>= 1.5`
- AWS CLI v2 with the Session Manager plugin
- AWS credentials with permission to manage EC2, IAM, SSM, and related Terraform
  resources under `deploy/terraform`

On macOS:

```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
brew install awscli
brew install --cask session-manager-plugin
```

Use a real deploy identity, not long-lived access keys committed to repo config.
Preferred setup is AWS IAM Identity Center/SSO or an IAM role that your local AWS
profile assumes. The deploy identity needs permission to read VPC/subnet/AMI
metadata, create the EC2 instance profile/role, pass that role to EC2, and
start/stop the instance.

Before deploying, verify that the active profile is the intended deploy identity:

```bash
aws sts get-caller-identity
```

If this returns a low-permission IAM user and Terraform fails with
`UnauthorizedOperation`, fix the AWS-side role/profile permissions first, then
rerun `make mlflow-deploy`.

## Provision Or Update

From the repository root:

```bash
make mlflow-deploy
```

This runs `terraform init` and `terraform apply` in `deploy/terraform`.

`user_data` runs only when an instance is first created. If you already deployed
an older MLflow instance and only change `user_data.sh.tftpl`, `terraform apply`
will not rewrite the running systemd service on that instance. Recreate the
instance intentionally, or update `/etc/systemd/system/mlflow.service` on the box
and restart `mlflow.service`.

## Start, Connect, Stop

Start the EC2 instance:

```bash
make mlflow-start
```

Open the SSM tunnel in a dedicated terminal and leave it running:

```bash
make mlflow-tunnel
```

Then set the tracking URI in the shell that runs experiments or backfills:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Open the same URL in a browser for the MLflow UI.

Stop the instance when idle:

```bash
make mlflow-stop
```

Stopping avoids EC2 compute charges while preserving the SQLite backend DB on
the EBS root volume. Destroying/replacing the instance loses that DB unless it
has been backed up; local JSON Run Records can be replayed with the backfill
script.

## Backfill Existing Records

Upload existing local Run Records without rerunning training. Backfill does not
provision infrastructure by itself; it only needs `MLFLOW_TRACKING_URI` to point
at a reachable tracking server. If the EC2 instance already exists, start it and
open the tunnel. Run `make mlflow-deploy` only when the infrastructure has not
been created yet.

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
make mlflow-backfill
```

For a sweep directory, upload only the newest immediate Hydra job directory:

```bash
make mlflow-backfill-latest MLFLOW_RECORDS=rawdata/logs/multirun/activationsearch
```

Limit the upload:

```bash
make mlflow-backfill MLFLOW_RECORDS=rawdata/logs/multirun/activationsearch
make mlflow-backfill MLFLOW_RECORDS=rawdata/logs/multirun/activationsearch/1069
```

Preview first:

```bash
make mlflow-backfill-dry-run
make mlflow-backfill-latest-dry-run MLFLOW_RECORDS=rawdata/logs/multirun/activationsearch
```

## Debug

```bash
aws ssm start-session --target <instance_id>

# on the instance:
systemctl status mlflow
journalctl -u mlflow -f
sqlite3 /opt/mlflow/mlflow.db .tables
```

## Tear Down

```bash
terraform -chdir=deploy/terraform destroy
```

Prefer `make mlflow-stop` for ordinary cost control. `terraform destroy` removes
the instance and loses the SQLite dashboard DB unless separately backed up.
