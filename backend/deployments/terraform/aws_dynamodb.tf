# ##############################################################################
# DynamoDB Table - Default
# ##############################################################################
resource "aws_dynamodb_table" "default" {
  name = "frappe-db-${terraform.workspace}-${var.aws_region}"

  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "PK"

  // Registrar ID
  attribute {
    name = "PK"
    type = "S"
  }

  tags = {
    Name = "FRAPPE - DynamoDB - ${terraform.workspace} - ${var.aws_region}"
  }
}
