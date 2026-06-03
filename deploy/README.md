# MLflow deployment

Self-hosted MLflow tracking server on AWS providing a **central comparison
dashboard** (params, metrics, tags). Full run data — the local JSON Run Records
and `result_<point>.pkl` curves/weights — stays on the training machine; S3
holds no result data today. The design and its trade-offs are recorded in
[`docs/adr/0001-self-hosted-mlflow-on-ec2.md`](../docs/adr/0001-self-hosted-mlflow-on-ec2.md).

```
your machine ──SSM tunnel──> EC2 (mlflow server, 127.0.0.1:5000)
                               ├─ backend store: sqlite on EBS
                               └─ Run Artifacts: s3://<bucket>  (via instance role)
client config: MLFLOW_TRACKING_URI=http://localhost:5000   (nothing else)
```

No inbound ports are opened: the server binds to localhost on the instance and
is reached only through an AWS SSM port-forwarding session. Artifacts are
**proxied** — the server writes to S3 using its IAM instance role, so no AWS
credentials ever live on a logging machine.

## Prerequisites

- Terraform `>= 1.5` and the AWS CLI v2 (with the Session Manager plugin).
- AWS credentials with permission to create S3, IAM, EC2, and SSM resources.

## Provision

```bash
cd deploy/terraform
terraform init
terraform apply          # review the plan, then confirm
```

Override defaults as needed, e.g.:

```bash
terraform apply -var region=eu-west-1 -var instance_type=t3.micro
```

Key outputs:

- `instance_id` — the SSM target.
- `artifact_bucket` — the S3 bucket for Run Artifacts.
- `ssm_port_forward_command` — a ready-to-run tunnel command.

## Connect

Open the tunnel (leave it running in its own terminal):

```bash
eval "$(terraform output -raw ssm_port_forward_command)"
```

Then, in your training environment:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
# open the UI at the same URL in a browser
```

> **Note:** the MLflow Run Record backend is **not implemented yet**. Today
> `src/experiment_logging.py` always writes local JSON; setting
> `MLFLOW_TRACKING_URI` has no effect until the adapter lands. The planned
> behavior (unset → local JSON, set → MLflow) is described in
> [`docs/adr/0002-mlflow-as-run-record-backend.md`](../docs/adr/0002-mlflow-as-run-record-backend.md).

## Cost control

Stop the instance when idle to avoid compute charges (EBS and S3 persist):

```bash
aws ec2 stop-instances  --instance-ids "$(terraform -chdir=deploy/terraform output -raw instance_id)"
aws ec2 start-instances --instance-ids "$(terraform -chdir=deploy/terraform output -raw instance_id)"
```

## Debug

```bash
# shell onto the box (no SSH key, no inbound port)
aws ssm start-session --target <instance_id>

# on the instance:
systemctl status mlflow
journalctl -u mlflow -f
sqlite3 /opt/mlflow/mlflow.db .tables
```

## Tear down

```bash
terraform destroy
```

The artifact bucket is created with `force_destroy = true`, so `terraform
destroy` empties it (including all object versions) and removes it cleanly. This
is safe because the bucket holds no source-of-truth data (see ADR-0001); drop
`force_destroy` if you ever store data there you want to protect from teardown.
