# ##############################################################################
# Lambdas
# We use aliases (named LIVE) to point to $LATEST as a mechanism for rolling 
# back deployments that go awry.
#
# Set the 'function_version' of the alias from $LATEST to the specific numbered 
# version and then rerun terraform. This has the least disruptive and quickest 
# rollback.
# ##############################################################################

# ##############################################################################
# Lambda - API
# ##############################################################################
resource "aws_lambda_function" "api" {
  function_name    = "frappe-api-${terraform.workspace}-${var.aws_region}"
  description      = var.build_version
  timeout          = 900
  memory_size      = 3008
  publish          = true
  role             = aws_iam_role.frappe_lambda_role.arn
  image_uri        = "${var.aws_account}.dkr.ecr.${var.aws_region}.amazonaws.com/frappe-lambda-api-${terraform.workspace}-${var.aws_region}:latest"
  source_code_hash = split(":", data.aws_ecr_image.lambda_api.image_digest)[1]
  package_type     = "Image"

  ephemeral_storage {
    size = 10240
  }

  tags = {
    Name = "FRAPPE - Lambda - API - ${terraform.workspace} - ${var.aws_region}"
  }
}

resource "aws_lambda_alias" "api_alias" {
  name             = "LIVE"
  description      = "FRAPPE API Current Release"
  function_name    = aws_lambda_function.api.arn
  function_version = "$LATEST"
}


resource "aws_lambda_permission" "api_invoke_by_gateway_default" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.default.execution_arn}/*/$default"
  qualifier     = "LIVE"
}

resource "aws_lambda_permission" "api_invoke_by_gateway_options" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.default.execution_arn}/*/*/{proxy+}"
  qualifier     = "LIVE"
}
