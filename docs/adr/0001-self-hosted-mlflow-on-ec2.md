---
status: accepted
---

# Self-hosted MLflow on EC2 with SQLite + S3, reached over an SSM tunnel

We will run the MLflow tracking server on a single small, stoppable EC2 instance
(`mlflow server --serve-artifacts --artifacts-destination s3://<bucket>
--backend-store-uri sqlite:///mlflow.db`), with run metadata in SQLite on EBS,
Run Artifacts proxied to S3 via the EC2's IAM instance role, and no internet
exposure — the server is reached only by SSM port-forwarding to
`http://localhost:5000`. We chose this over managed and production-grade
alternatives because the project is a single-researcher thesis where cost and
debuggability outweigh managed convenience and scale, and because durable
central storage (not multi-user serving) is the only stage-1 requirement.

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

- A future move to running training *on* AWS compute (a later stage) needs no
  code change: in-VPC jobs reach the same server privately, and the client
  contract stays "set `MLFLOW_TRACKING_URI`".
- If concurrent logging from many parallel AWS jobs ever materializes, SQLite's
  write concurrency becomes the first thing to revisit (migrate backend to
  Postgres/RDS) — this ADR would be superseded at that point.
- Infrastructure is provisioned by a Terraform module under `deploy/terraform/`
  (S3 + IAM + EC2 + security group + SSM) so the topology is codified and
  teardown/recreate is reproducible.
