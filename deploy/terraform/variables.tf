variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "eu-central-1"
}

variable "project" {
  description = "Name prefix and tag applied to all resources."
  type        = string
  default     = "nnforhjb"
}

variable "instance_type" {
  description = "EC2 instance type for the MLflow server. t3.small is ample for a single sequential logger."
  type        = string
  default     = "t3.small"
}

variable "artifact_bucket_prefix" {
  description = "Prefix for the S3 artifact bucket; a random suffix is appended for global uniqueness."
  type        = string
  default     = "nnforhjb-mlflow"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB. Holds the SQLite backend store, so size for run-metadata growth."
  type        = number
  default     = 20
}

variable "mlflow_port" {
  description = "Port the MLflow server listens on (bound to localhost; reached via SSM port-forward)."
  type        = number
  default     = 5000
}

variable "mlflow_version" {
  description = "MLflow version to pip-install. Empty string installs the latest release."
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Subnet to launch into. Defaults to the first subnet of the account's default VPC."
  type        = string
  default     = null
}
