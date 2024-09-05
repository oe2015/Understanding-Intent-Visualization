# ##############################################################################
# ECR - BASE-LAMBDA
# ##############################################################################
resource "aws_ecr_repository" "base_lambda" {
  name                 = "frappe-base-lambda-${terraform.workspace}-${var.aws_region}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "base_lambda" {
  repository = aws_ecr_repository.base_lambda.name

  policy = templatefile("policies/aws_ecr_lifecycle_policy.json", {})
}
