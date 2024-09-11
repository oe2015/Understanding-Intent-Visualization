# ##############################################################################
# Lambda
# ##############################################################################
resource "aws_iam_role" "frappe_lambda_role" {
  name = "frappe-lambda-role-${terraform.workspace}-${var.aws_region}"

  tags = {
    Name = "IAM Role - FRAPPE - Lambda - ${terraform.workspace} - ${var.aws_region}"
  }

  assume_role_policy = templatefile("policies/aws_lambda_role.json", {})
}

resource "aws_iam_policy" "frappe_lambda_policy" {
  name        = "frappe-lambda-policy-${terraform.workspace}-${var.aws_region}"
  path        = "/"
  description = "Policy for the frappe Lambdas"

  policy = templatefile("policies/aws_lambda_policy.json", {
    dynamodb_arn     = aws_dynamodb_table.default.arn
    lambda_task1_arn = aws_lambda_function.task1.arn
    lambda_task2_arn = aws_lambda_function.task2.arn
    lambda_task3_arn = aws_lambda_function.task3.arn
  })
}

resource "aws_iam_role_policy_attachment" "frappe_lambda_role_policy" {
  role       = aws_iam_role.frappe_lambda_role.name
  policy_arn = aws_iam_policy.frappe_lambda_policy.arn
}
