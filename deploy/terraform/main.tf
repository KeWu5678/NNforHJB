# ---------------------------------------------------------------------------
# MLflow tracking server: one stoppable EC2 instance, SQLite backend on EBS,
# reached only over SSM.
# See docs/adr/0001-self-hosted-mlflow-on-ec2.md.
# ---------------------------------------------------------------------------

# --- Networking ---
# When var.subnet_id is set, derive everything from that subnet's VPC (no
# default VPC required); otherwise fall back to the account's default VPC. The
# security group is created in whichever VPC the instance lands in, so a custom
# subnet in a non-default VPC works.
data "aws_vpc" "default" {
  count   = var.subnet_id == null ? 1 : 0
  default = true
}

data "aws_subnets" "default" {
  count = var.subnet_id == null ? 1 : 0
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default[0].id]
  }
}

data "aws_subnet" "selected" {
  count = var.subnet_id == null ? 0 : 1
  id    = var.subnet_id
}

locals {
  subnet_id = var.subnet_id != null ? var.subnet_id : data.aws_subnets.default[0].ids[0]
  vpc_id    = var.subnet_id != null ? data.aws_subnet.selected[0].vpc_id : data.aws_vpc.default[0].id
}

# --- Latest Amazon Linux 2023 AMI (SSM Agent preinstalled) ---
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

# ---------------------------------------------------------------------------
# Instance role: SSM access only
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "mlflow" {
  name               = "${var.project}-mlflow-ec2"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

# Enables SSM Session Manager / port-forwarding (no inbound SSH needed).
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.mlflow.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "mlflow" {
  name = "${var.project}-mlflow"
  role = aws_iam_role.mlflow.name
}

# ---------------------------------------------------------------------------
# Security group: no inbound (SSM only), all outbound (SSM, package installs)
# ---------------------------------------------------------------------------

resource "aws_security_group" "mlflow" {
  name        = "${var.project}-mlflow"
  description = "MLflow server: no inbound, reached via SSM only"
  vpc_id      = local.vpc_id

  egress {
    description = "All outbound (SSM endpoints, package installs)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------

resource "aws_instance" "mlflow" {
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.instance_type
  subnet_id              = local.subnet_id
  iam_instance_profile   = aws_iam_instance_profile.mlflow.name
  vpc_security_group_ids = [aws_security_group.mlflow.id]

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    mlflow_port    = var.mlflow_port
    mlflow_version = var.mlflow_version
  })

  # Do NOT replace the instance when the bootstrap script changes: replacement
  # would destroy the SQLite backend store on the root volume. Re-apply
  # bootstrap changes manually (re-run on the box, or `terraform taint` to opt
  # in explicitly). See docs/adr/0001-self-hosted-mlflow-on-ec2.md.
  user_data_replace_on_change = false

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  tags = {
    Name = "${var.project}-mlflow"
  }
}
