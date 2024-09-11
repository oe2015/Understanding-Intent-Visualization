# ##############################################################################
# ECR - API
# ##############################################################################
resource "aws_ecr_repository" "lambda_api" {
  name                 = "frappe-lambda-api-${terraform.workspace}-${var.aws_region}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "lambda_api" {
  repository = aws_ecr_repository.lambda_api.name

  policy = templatefile("policies/aws_ecr_lifecycle_policy.json", {})
}

# ##############################################################################
# ECR - Task1
# ##############################################################################
resource "aws_ecr_repository" "lambda_task1" {
  name                 = "frappe-lambda-task1-${terraform.workspace}-${var.aws_region}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "lambda_task1" {
  repository = aws_ecr_repository.lambda_task1.name

  policy = templatefile("policies/aws_ecr_lifecycle_policy.json", {})
}

# ##############################################################################
# ECR - Task2
# ##############################################################################
resource "aws_ecr_repository" "lambda_task2" {
  name                 = "frappe-lambda-task2-${terraform.workspace}-${var.aws_region}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "lambda_task2" {
  repository = aws_ecr_repository.lambda_task2.name

  policy = templatefile("policies/aws_ecr_lifecycle_policy.json", {})
}

# ##############################################################################
# ECR - Task3
# ##############################################################################
resource "aws_ecr_repository" "lambda_task3" {
  name                 = "frappe-lambda-task3-${terraform.workspace}-${var.aws_region}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "lambda_task3" {
  repository = aws_ecr_repository.lambda_task3.name

  policy = templatefile("policies/aws_ecr_lifecycle_policy.json", {})
}
