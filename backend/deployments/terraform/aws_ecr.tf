# ##############################################################################
# ECR Image - API
# ##############################################################################
data "aws_ecr_image" "lambda_api" {
  repository_name = "frappe-lambda-api-${terraform.workspace}-${var.aws_region}"
  image_tag       = "latest"
}
