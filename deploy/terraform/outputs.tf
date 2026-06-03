output "instance_id" {
  description = "EC2 instance ID — the SSM port-forward target."
  value       = aws_instance.mlflow.id
}

output "artifact_bucket" {
  description = "S3 bucket holding MLflow Run Artifacts."
  value       = aws_s3_bucket.artifacts.id
}

output "mlflow_port" {
  description = "Port the server listens on (localhost on the instance)."
  value       = var.mlflow_port
}

output "ssm_port_forward_command" {
  description = "Run this to open the tunnel, then set MLFLOW_TRACKING_URI=http://localhost:<mlflow_port>."
  value = join(" ", [
    "aws ssm start-session",
    "--region ${var.region}",
    "--target ${aws_instance.mlflow.id}",
    "--document-name AWS-StartPortForwardingSession",
    "--parameters '{\"portNumber\":[\"${var.mlflow_port}\"],\"localPortNumber\":[\"${var.mlflow_port}\"]}'",
  ])
}
