# ##############################################################################
# ECR - ICD10-CM
# ##############################################################################
resource "aws_ecr_repository" "lambda_icd10cm" {
  name                 = "frappe-lambda-api-${terraform.workspace}-${var.aws_region}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "lambda_icd10cm" {
  repository = aws_ecr_repository.lambda_icd10cm.name

  policy = templatefile("policies/aws_ecr_lifecycle_policy.json", {})
}
