resource "aws_apigatewayv2_api" "default" {
  name          = "frappe-api-${terraform.workspace}-${var.aws_region}"
  description   = "FRAPPE API Gateway for ${terraform.workspace}-${var.aws_region}"
  protocol_type = "HTTP"

  disable_execute_api_endpoint = false

  tags = {
    Name = "APIGateway - FRAPPE - API - ${terraform.workspace} - ${var.aws_region}"
  }
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.default.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api.arn
    format          = <<JSON
  { "requestTime": "$context.requestTime", "requestId": "$context.requestId", "httpMethod": "$context.httpMethod", "path": "$context.path", "routeKey": "$context.routeKey", "status": $context.status, "responseLatency": $context.responseLatency, "integrationRequestId": "$context.integration.requestId", "functionResponseStatus": "$context.integration.status", "integrationLatency": "$context.integration.latency", "integrationServiceStatus": "$context.integration.integrationStatus", "authorizeResultStatus": "$context.authorizer.status", "authorizerRequestId": "$context.authorizer.requestId", "ip": "$context.identity.sourceIp", "userAgent": "$context.identity.userAgent", "principalId": "$context.authorizer.principalId", "error": {   "message": "$context.error.message",   "responseType": "$context.error.responseType" } }
  JSON
  }

  tags = {
    Name = "APIGateway Stage - FRAPPE - API - ${terraform.workspace} - ${var.aws_region}"
  }
}

resource "aws_apigatewayv2_integration" "default" {
  api_id      = aws_apigatewayv2_api.default.id
  description = "FRAPPE API Lambda for ${terraform.workspace}-${var.aws_region}"

  integration_type       = "AWS_PROXY"
  connection_type        = "INTERNET"
  integration_method     = "POST"
  integration_uri        = aws_lambda_alias.api_alias.invoke_arn
  passthrough_behavior   = "WHEN_NO_MATCH"
  payload_format_version = "2.0"
}

# resource "aws_apigatewayv2_authorizer" "default" {
#   api_id           = aws_apigatewayv2_api.default.id
#   authorizer_type  = "JWT"
#   identity_sources = ["$request.header.Authorization"]
#   name             = "jwt-cognito-${terraform.workspace}-${var.aws_region}"

#   jwt_configuration {
#     audience = [lookup(local.jwt, terraform.workspace, local.jwt["default"]).audience]
#     issuer   = lookup(local.jwt, terraform.workspace, local.jwt["default"]).issuer_uri
#   }
# }

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.default.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.default.id}"
  //authorization_type = "JWT"
  #authorizer_id      = aws_apigatewayv2_authorizer.default.id
}

resource "aws_apigatewayv2_route" "options" {
  api_id             = aws_apigatewayv2_api.default.id
  route_key          = "OPTIONS /{proxy+}"
  target             = "integrations/${aws_apigatewayv2_integration.default.id}"
  authorization_type = "NONE"
}

# resource "aws_apigatewayv2_domain_name" "default" {
#   domain_name = local.api_domain

#   domain_name_configuration {
#     certificate_arn = local.certificate_arn
#     endpoint_type   = "REGIONAL"
#     security_policy = "TLS_1_2"
#   }

#   tags = {
#     Name = "APIGateway Domain - FRAPPE - API - ${terraform.workspace} - ${var.aws_region}"
#   }
# }

# resource "aws_apigatewayv2_api_mapping" "default" {
#   api_id      = aws_apigatewayv2_api.default.id
#   domain_name = aws_apigatewayv2_domain_name.default.id
#   stage       = aws_apigatewayv2_stage.default.id
# }

# ##############################################################################
# API Gateway Logs
# ##############################################################################
resource "aws_cloudwatch_log_group" "api" {
  name              = "/frappe-${terraform.workspace}-${var.aws_region}/api/"
  retention_in_days = 30

  tags = {
    Name = "Cloudwatch - FRAPPE - API - ${terraform.workspace} - ${var.aws_region}"
  }
}
