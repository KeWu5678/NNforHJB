---
status: proposed
---

# Self-hosted MLflow on EC2 with SQLite + S3, reached over an SSM tunnel

We will run the MLflow tracking server on a single small, stoppable EC2 instance
(`mlflow server --serve-artifacts --artifacts-destination s3://<bucket>
--backend-store-uri sqlite:///mlflow.db`), with run metadata in SQLite on EBS,
Run Artifacts proxied to S3 via the EC2's IAM instance role, and no internet
exposure — the server binds to localhost on the instance and is reached only by
SSM port-forwarding to `http://localhost:5000`. We chose this over managed and
production-grade alternatives because the project is a single-researcher thesis
where cost and debuggability outweigh managed convenience and scale, and because
the stage-1 requirement is a central comparison dashboard (not multi-user
serving) — full run data stays local, as detailed under "What lives where"
below.

**Implementation status: proposed, not yet deployed.** Terraform for this
topology lives on the `mlflow-deployment-plan` branch (`deploy/terraform/`) but
has not been applied or validated against AWS. Flip this ADR to `accepted` once
the module is applied and the server is reachable.

## Considered options

- **SageMaker managed MLflow** — rejected: ~$0.64/hr while running, a black box
  to debug, and SageMaker lock-in, none of which a single user needs.
- **ECS Fargate + RDS Postgres + S3** — rejected: production-grade and more
  durable, but ~$40–60/mo always-on and many moving parts (ALB, RDS, VPC, IAM)
  to debug for a workload that is one sequential logger.
- **Public endpoint + IP allowlist + basic auth + TLS** — rejected: "view UI
  from anywhere / share" was explicitly *not* a stage-1 driver, so the TLS-cert
  and password machinery buys nothing; the SSM tunnel removes all inbound
  exposure instead.
- **Proxied artifacts (chosen) vs direct-to-S3** — proxied so the client needs
  only `MLFLOW_TRACKING_URI`; no AWS credentials ever live on a logging machine.
- **SQLite (chosen) vs Postgres backend** — SQLite is sufficient for a single
  sequential logger; Postgres would be a database to run, back up, and debug for
  no benefit at this scale.

## Consequences

### What lives where (no single "durable central" tier)

- **Local, on the training machine — the source of truth.** The local JSON Run
  Records and the full `result_<point>.pkl` (per-iteration curves + weights) are
  written next to the run and are **not** copied to S3. They are durable on that
  machine but are *not* centrally durable; per
  [ADR-0002](0002-mlflow-as-run-record-backend.md) the pickle is referenced from
  MLflow only by a `result_pickle` tag (a pointer), not stored centrally.
- **S3 artifact store — only what MLflow explicitly logs.** The server is
  configured to proxy artifacts to S3, but the current adapter logs only params,
  metrics, and tags — **no artifacts** — so S3 holds no result data today. It
  covers only files passed to `mlflow.log_artifact` (none yet; future plots, if
  any). S3 itself is durable across `terraform destroy`.
- **Central but instance-lifecycle:** the MLflow summary metrics/params/tags in
  the SQLite backend store on the EBS root volume. This is what makes the
  comparison dashboard central. It survives stop/start, but is **lost on instance
  replacement or `terraform destroy`** — treat it as a rebuildable index, not the
  source of truth. If that history must survive a destroy/recreate, that is a
  separate decision (periodic SQLite → S3 backup, or a retained data volume).
- **Net:** what is centrally available is the *comparison summary*, not the full
  results. Full results stay local by choice; this narrows the original
  "durable central store" goal to "central comparison + local results."
- The Terraform sets `user_data_replace_on_change = false` so that editing the
  bootstrap script does not silently replace the instance and discard the SQLite
  store; bootstrap changes are re-applied manually (or by an explicit taint).
- "Reproducible teardown/recreate" therefore refers to the **infrastructure**
  (Terraform rebuilds S3 + IAM + EC2 + SG identically), not to the SQLite
  metadata, which is intentionally ephemeral.

### Stage-3 (training on AWS) reachability

- The **client contract is unchanged** for a future move to running training on
  AWS compute: jobs still only need `MLFLOW_TRACKING_URI`.
- **Reaching** the endpoint from inside the VPC is *not* free, though: with a
  localhost bind and no inbound security-group rule, an in-VPC job must either
  SSM-port-forward like the laptop does, or the topology must add a private bind
  plus a security-group rule scoped to the VPC/job. That is a small infra
  revision (no application code change), to be recorded when stage 3 lands.

### Other

- If concurrent logging from many parallel AWS jobs ever materializes, SQLite's
  write concurrency becomes the first thing to revisit (migrate backend to
  Postgres/RDS) — this ADR would be superseded at that point.
- Infrastructure is provisioned by a Terraform module under `deploy/terraform/`
  (S3 + IAM + EC2 + security group + SSM).
