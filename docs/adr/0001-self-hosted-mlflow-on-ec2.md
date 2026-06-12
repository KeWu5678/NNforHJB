---
status: proposed
---

# Self-hosted MLflow on EC2 with SQLite on EBS, reached over an SSM tunnel

We will run the MLflow tracking server on a single small, stoppable EC2 instance
with a SQLite backend store on the instance's EBS root volume. The server is not
publicly exposed; it is reached through an SSM port-forwarding tunnel to
`http://localhost:5000`.

This topology is for a single-researcher thesis workflow. The goal is a central
comparison dashboard while active, not a fully managed durable artifact service.
Stopping and starting the same EC2 instance preserves the SQLite DB on EBS, so
the MLflow UI keeps previous runs without paying for compute while stopped.

**Implementation status: proposed, not yet deployed.** Terraform lives under
`deploy/terraform/`; v1 application logging does not require S3 and does not
upload artifacts.

## Decision

- Use EC2 + SQLite on EBS for the MLflow backend store.
- Reach the server only through SSM port forwarding; no public inbound MLflow
  endpoint is needed for v1.
- Preserve dashboard history by stopping/starting the same EC2 instance, not by
  destroying/replacing it.
- Do not require an S3 artifact bucket in v1. Local Run Records and result
  pickles remain the source of truth; MLflow records only dashboard metadata and
  local artifact pointers, per [ADR-0002](0002-mlflow-as-run-record-backend.md).

## Considered Options

- **SageMaker managed MLflow** — rejected: higher always-on cost and more managed
  machinery than a single-user research dashboard needs.
- **ECS Fargate + RDS Postgres + S3** — rejected: production-grade but excessive
  for one sequential logger.
- **Public endpoint + IP allowlist + basic auth + TLS** — rejected: stage 1 does
  not require sharing the UI from arbitrary networks.
- **S3 artifact store in v1** — rejected for now. The current workflow keeps full
  result artifacts locally and uses MLflow only for dashboard metadata.
- **SQLite on EBS** — chosen because it is simple, cheap while stopped, and
  sufficient for a single sequential logger.

## Consequences

- While the EC2 instance is stopped, compute cost is avoided but EBS storage cost
  remains.
- If the instance and EBS volume are preserved, starting MLflow again shows prior
  runs from the same SQLite DB.
- If the instance/backend DB is destroyed, S3 artifacts alone would not rebuild
  the MLflow dashboard. Recovery requires a DB backup/restore or replaying local
  JSON Run Records with `scripts/upload_run_records_to_mlflow.py`.
- If concurrent AWS training jobs or shared remote artifact downloads become
  important, revisit the backend store and artifact-store decisions.
