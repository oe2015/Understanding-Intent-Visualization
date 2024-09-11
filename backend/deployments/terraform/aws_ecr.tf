# ##############################################################################
# ECR Image - API
# ##############################################################################
data "aws_ecr_image" "lambda_api" {
  repository_name = "frappe-lambda-api-${terraform.workspace}-${var.aws_region}"
  image_tag       = "latest"
}

# ##############################################################################
# ECR Image - Task1
# ##############################################################################
data "aws_ecr_image" "lambda_task1" {
  repository_name = "frappe-lambda-task1-${terraform.workspace}-${var.aws_region}"
  image_tag       = "latest"
}

# ##############################################################################
# ECR Image - Task2
# ##############################################################################
data "aws_ecr_image" "lambda_task2" {
  repository_name = "frappe-lambda-task2-${terraform.workspace}-${var.aws_region}"
  image_tag       = "latest"
}

# ##############################################################################
# ECR Image - Task3
# ##############################################################################
data "aws_ecr_image" "lambda_task3" {
  repository_name = "frappe-lambda-task3-${terraform.workspace}-${var.aws_region}"
  image_tag       = "latest"
}

